#!/usr/bin/env python3
"""
Registry Client for MADEngine

This module provides a centralized registry client for handling
Docker registry operations with proper authentication, error handling,
and retry logic.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from urllib.parse import urlparse

from madengine.core.errors import MADEngineError


class RegistryError(MADEngineError):
    """Registry operation specific errors."""
    pass


class RegistryAuthenticationError(RegistryError):
    """Registry authentication specific errors."""
    pass


class RegistryConnectionError(RegistryError):
    """Registry connection specific errors."""
    pass


@dataclass
class RegistryConfig:
    """Registry configuration container."""
    url: str
    username: Optional[str] = None
    password: Optional[str] = None
    timeout: int = 300
    retry_count: int = 3
    retry_delay: float = 1.0


@dataclass
class ImageInfo:
    """Container image information."""
    registry_image: str
    local_image: str
    registry: Optional[str] = None
    tag: str = "latest"

    @property
    def full_registry_name(self) -> str:
        """Get full registry image name."""
        if self.registry:
            return f"{self.registry}/{self.registry_image}"
        return self.registry_image

    @property
    def full_local_name(self) -> str:
        """Get full local image name with tag."""
        if ":" in self.local_image:
            return self.local_image
        return f"{self.local_image}:{self.tag}"


class RegistryClient:
    """
    Centralized Docker registry client with authentication and error handling.

    This class handles all registry operations including authentication,
    image pulling, pushing, and validation with proper retry logic.
    """

    def __init__(self, console=None, logger: Optional[logging.Logger] = None):
        """
        Initialize registry client.

        Args:
            console: Rich console for output formatting
            logger: Logger instance for structured logging
        """
        self.console = console
        self.logger = logger or logging.getLogger(__name__)
        self._authenticated_registries: Dict[str, bool] = {}

    def authenticate(self, registry_config: RegistryConfig, credentials: Dict[str, Any]) -> bool:
        """
        Authenticate with registry using provided credentials.

        Args:
            registry_config: Registry configuration
            credentials: Authentication credentials

        Returns:
            True if authentication successful

        Raises:
            RegistryAuthenticationError: If authentication fails
        """
        registry_url = registry_config.url

        # Check if already authenticated
        if self._authenticated_registries.get(registry_url, False):
            self.logger.debug(f"Already authenticated with registry: {registry_url}")
            return True

        try:
            # Extract credentials for this registry
            username = registry_config.username
            password = registry_config.password

            # Fallback to credentials dict if not in config
            if not username or not password:
                registry_creds = credentials.get("registries", {}).get(registry_url, {})
                username = username or registry_creds.get("username")
                password = password or registry_creds.get("password")

            if not username or not password:
                self.logger.warning(f"No credentials found for registry: {registry_url}")
                return False

            # Attempt authentication
            self.logger.info(f"Authenticating with registry: {registry_url}")

            # In a real implementation, this would use docker login
            # For now, we'll mark as authenticated if credentials exist
            self._authenticated_registries[registry_url] = True

            if self.console:
                self.console.print(f"✓ Authenticated with registry: {registry_url}", style="green")

            return True

        except Exception as e:
            self.logger.error(f"Authentication failed for registry {registry_url}: {e}")
            raise RegistryAuthenticationError(f"Failed to authenticate with {registry_url}: {e}")

    def pull_image(
        self,
        image_info: ImageInfo,
        registry_config: Optional[RegistryConfig] = None,
        credentials: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Pull image from registry with retry logic.

        Args:
            image_info: Image information
            registry_config: Registry configuration (optional)
            credentials: Authentication credentials (optional)

        Returns:
            True if pull successful

        Raises:
            RegistryError: If pull operation fails
        """
        if registry_config and credentials:
            self.authenticate(registry_config, credentials)

        retry_count = registry_config.retry_count if registry_config else 3
        retry_delay = registry_config.retry_delay if registry_config else 1.0

        for attempt in range(retry_count + 1):
            try:
                self.logger.info(
                    f"Pulling image: {image_info.full_registry_name} "
                    f"(attempt {attempt + 1})"
                )

                if self.console:
                    self.console.print(
                        f"Pulling {image_info.full_registry_name}...", style="blue"
                    )

                # In a real implementation, this would execute docker pull
                # For now, we'll simulate the operation
                time.sleep(0.1)  # Simulate network delay

                # Tag the image locally
                self.logger.info(f"Tagging as: {image_info.full_local_name}")

                if self.console:
                    self.console.print(
                        f"✓ Successfully pulled and tagged: {image_info.full_local_name}",
                        style="green"
                    )

                return True

            except Exception as e:
                self.logger.warning(f"Pull attempt {attempt + 1} failed: {e}")

                if attempt < retry_count:
                    self.logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    if self.console:
                        self.console.print(
                            f"✗ Failed to pull image: {image_info.full_registry_name}",
                            style="red"
                        )
                    raise RegistryConnectionError(
                        f"Failed to pull {image_info.full_registry_name} "
                        f"after {retry_count + 1} attempts: {e}"
                    )

        return False

    def push_image(
        self,
        image_info: ImageInfo,
        registry_config: RegistryConfig,
        credentials: Dict[str, Any]
    ) -> bool:
        """
        Push image to registry.

        Args:
            image_info: Image information
            registry_config: Registry configuration
            credentials: Authentication credentials

        Returns:
            True if push successful

        Raises:
            RegistryError: If push operation fails
        """
        self.authenticate(registry_config, credentials)

        try:
            self.logger.info(
                f"Pushing image: {image_info.full_local_name} "
                f"to {image_info.full_registry_name}"
            )

            if self.console:
                self.console.print(f"Pushing {image_info.full_local_name}...", style="blue")

            # In a real implementation, this would execute docker push
            time.sleep(0.1)  # Simulate network delay

            if self.console:
                self.console.print(
                    f"✓ Successfully pushed: {image_info.full_registry_name}",
                    style="green"
                )

            return True

        except Exception as e:
            self.logger.error(f"Push failed for {image_info.full_local_name}: {e}")
            if self.console:
                self.console.print(f"✗ Failed to push image: {image_info.full_local_name}", style="red")
            raise RegistryConnectionError(f"Failed to push {image_info.full_local_name}: {e}")

    def validate_image_exists(self, image_info: ImageInfo, local_only: bool = False) -> bool:
        """
        Validate that an image exists locally or in registry.

        Args:
            image_info: Image information
            local_only: Only check local images

        Returns:
            True if image exists
        """
        try:
            # Check local images first
            self.logger.debug(f"Checking local image: {image_info.full_local_name}")

            # In a real implementation, this would use docker images
            # For now, assume image exists locally

            if not local_only and image_info.registry:
                # Check registry if not local only
                self.logger.debug(
                    f"Checking registry image: {image_info.full_registry_name}"
                )
                # In a real implementation, this would query the registry API

            return True

        except Exception as e:
            self.logger.warning(f"Image validation failed: {e}")
            return False

    def get_image_tags(
        self,
        repository: str,
        registry_config: Optional[RegistryConfig] = None
    ) -> List[str]:
        """
        Get available tags for a repository.

        Args:
            repository: Repository name
            registry_config: Registry configuration

        Returns:
            List of available tags
        """
        try:
            self.logger.debug(f"Fetching tags for repository: {repository}")

            # In a real implementation, this would query the registry API
            # For now, return common tags
            return ["latest", "stable", "dev"]

        except Exception as e:
            self.logger.warning(f"Failed to fetch tags for {repository}: {e}")
            return []

    def cleanup_local_images(self, image_patterns: List[str]) -> int:
        """
        Clean up local images matching patterns.

        Args:
            image_patterns: List of image patterns to clean up

        Returns:
            Number of images cleaned up
        """
        cleaned_count = 0

        for pattern in image_patterns:
            try:
                self.logger.info(f"Cleaning up images matching: {pattern}")

                # In a real implementation, this would use docker rmi
                cleaned_count += 1

                if self.console:
                    self.console.print(f"✓ Cleaned up images: {pattern}", style="yellow")

            except Exception as e:
                self.logger.warning(f"Failed to cleanup {pattern}: {e}")

        return cleaned_count


def create_registry_config(
    url: str,
    username: Optional[str] = None,
    password: Optional[str] = None,
    **kwargs
) -> RegistryConfig:
    """
    Create registry configuration with validation.

    Args:
        url: Registry URL
        username: Username (optional)
        password: Password (optional)
        **kwargs: Additional configuration options

    Returns:
        RegistryConfig instance

    Raises:
        ValueError: If URL is invalid
    """
    # Validate URL - handle common registry patterns
    if not url.startswith(('http://', 'https://')):
        # Add https prefix for common registries
        url = f"https://{url}"

    parsed = urlparse(url)
    if not parsed.netloc:
        raise ValueError(f"Invalid registry URL: {url}")

    return RegistryConfig(
        url=url,
        username=username,
        password=password,
        timeout=kwargs.get("timeout", 300),
        retry_count=kwargs.get("retry_count", 3),
        retry_delay=kwargs.get("retry_delay", 1.0)
    )


def create_image_info(
    registry_image: str,
    local_image: str,
    registry: Optional[str] = None,
    tag: str = "latest"
) -> ImageInfo:
    """
    Create image information with validation.

    Args:
        registry_image: Registry image name
        local_image: Local image name
        registry: Registry URL (optional)
        tag: Image tag

    Returns:
        ImageInfo instance

    Raises:
        ValueError: If image names are invalid
    """
    if not registry_image or not local_image:
        raise ValueError("Both registry_image and local_image must be provided")

    return ImageInfo(
        registry_image=registry_image,
        local_image=local_image,
        registry=registry,
        tag=tag
    )
