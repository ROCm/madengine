"""Wheel indexes and wheel metadata.

Framework wheel indexes (TheRock, rocm.frameworks.amd.com) serve PEP 503 pages,
sometimes nested one level per stack (``.../whl-multi-arch/vllm/amd-aiter/``),
without PEP 658 metadata or hashes. Metadata is therefore read from the wheel
itself with HTTP range requests: a wheel is a zip, so only the central
directory and ``METADATA`` are fetched, a few KB out of a multi-GB file.
"""

import json
import os
import re
import typing
import urllib.parse
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser

USER_AGENT = "madengine-stack"
_WHEEL_RE = re.compile(
    r"^(?P<name>[^-]+)-(?P<version>[^-]+)(?:-(?P<build>\d[^-]*))?"
    r"-(?P<python>[^-]+)-(?P<abi>[^-]+)-(?P<platform>[^-]+)\.whl$"
)


@dataclass
class Wheel:
    """One wheel file in an index."""

    name: str
    version: str
    url: str
    python: str
    platform: str
    index: str = ""
    local: typing.Dict[str, str] = field(default_factory=dict)
    size: typing.Optional[int] = None
    etag: typing.Optional[str] = None
    requires: typing.Optional[typing.List[str]] = None
    requires_python: typing.Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v not in (None, {}, "")}


def normalize(name: str) -> str:
    """PEP 503 project name normalization."""
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_local(version: str) -> typing.Dict[str, str]:
    """Provenance from a local version: ``+rocm10.1.0.g98dff2a81.d20261002``."""
    if "+" not in version:
        return {}
    out = {}
    for part in version.split("+", 1)[1].split("."):
        if part.startswith("rocm"):
            out["rocm"] = part[4:]
        elif re.fullmatch(r"g[0-9a-f]{7,40}", part):
            out["git"] = part[1:]
        elif re.fullmatch(r"d\d{8}", part):
            out["date"] = part[1:]
        elif out.get("rocm") is not None and re.fullmatch(r"\d+", part):
            out["rocm"] += "." + part
    return out


def parse_wheel_filename(url: str) -> typing.Optional[Wheel]:
    filename = urllib.parse.unquote(url.rsplit("/", 1)[-1].split("#", 1)[0])
    m = _WHEEL_RE.match(filename)
    if not m:
        return None
    version = m.group("version")
    return Wheel(
        name=normalize(m.group("name")),
        version=version,
        url=url.split("#", 1)[0],
        python=m.group("python"),
        platform=m.group("platform"),
        local=parse_local(version),
    )


def _get(url: str, method: str = "GET", headers: typing.Optional[dict] = None):
    req = urllib.request.Request(
        url, method=method, headers={"User-Agent": USER_AGENT, **(headers or {})}
    )
    return urllib.request.urlopen(req, timeout=60)


class _Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.hrefs.append(href)


def list_links(url: str) -> typing.List[str]:
    """Absolute hrefs on a PEP 503 page (after redirects)."""
    with _get(url if url.endswith("/") else url + "/") as resp:
        base = resp.geturl()
        parser = _Links()
        parser.feed(resp.read().decode("utf-8", "replace"))
    return [urllib.parse.urljoin(base, h) for h in parser.hrefs]


def walk_index(
    url: str,
    projects: typing.Optional[typing.Iterable[str]] = None,
    max_depth: int = 2,
) -> typing.List[Wheel]:
    """All wheels under an index, following nested per-stack indexes.

    A link that ends in ``/`` is a project page or a nested index; its wheels
    are tagged with the path of the index they came from (``vllm-cdna``), which
    is how per-stack builds of the same component stay distinguishable.
    """
    wanted = {normalize(p) for p in projects} if projects else None
    root = url if url.endswith("/") else url + "/"
    wheels: typing.List[Wheel] = []

    def visit(page: str, depth: int) -> None:
        for link in list_links(page):
            if link.endswith(".whl") or ".whl#" in link:
                wheel = parse_wheel_filename(link)
                if wheel and (wanted is None or wheel.name in wanted):
                    rel = page[len(root):].strip("/").split("/")
                    wheel.index = "/".join(rel[:-1])  # drop the project segment
                    wheels.append(wheel)
            elif link.endswith("/") and link.startswith(page) and depth < max_depth:
                visit(link, depth + 1)

    visit(root, 0)
    return wheels


class _RangeFile:
    """Seekable read-only view of a remote file over HTTP range requests."""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        with _get(url, method="HEAD") as resp:
            self.size = int(resp.headers["Content-Length"])
            self.etag = resp.headers.get("ETag")

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def read(self, n=-1):
        end = self.size - 1 if n is None or n < 0 else min(self.size - 1, self.pos + n - 1)
        if self.pos > end:
            return b""
        with _get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}) as resp:
            data = resp.read()
        self.pos += len(data)
        return data


def _cache_dir() -> str:
    root = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache"
    )
    return os.path.join(root, "madengine", "wheel-metadata")


def fetch_metadata(wheel: Wheel, use_cache: bool = True) -> Wheel:
    """Fill size, etag, requires and requires_python from the remote wheel.

    Cached by ETag, so a republished file under the same name is re-read.
    """
    remote = _RangeFile(wheel.url)
    wheel.size, wheel.etag = remote.size, remote.etag
    key = re.sub(r"[^A-Za-z0-9]", "", (remote.etag or "") + str(remote.size))
    cache = os.path.join(_cache_dir(), f"{wheel.name}-{key}.json")
    if use_cache and remote.etag and os.path.exists(cache):
        with open(cache) as f:
            cached = json.load(f)
        wheel.requires = cached["requires"]
        wheel.requires_python = cached.get("requires_python")
        return wheel

    archive = zipfile.ZipFile(remote)
    meta = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
    lines = archive.read(meta).decode("utf-8", "replace").splitlines()
    wheel.requires = [l[len("Requires-Dist:"):].strip()
                      for l in lines if l.startswith("Requires-Dist:")]
    wheel.requires_python = next(
        (l.split(":", 1)[1].strip() for l in lines if l.startswith("Requires-Python:")),
        None,
    )
    if remote.etag:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, "w") as f:
            json.dump({"requires": wheel.requires,
                       "requires_python": wheel.requires_python}, f)
    return wheel


def head(url: str) -> typing.Dict[str, typing.Any]:
    """Size and ETag of a remote file, for drift checks."""
    with _get(url, method="HEAD") as resp:
        return {"size": int(resp.headers.get("Content-Length", 0)),
                "etag": resp.headers.get("ETag")}
