# Kubeconfig Symlink Setup ✅

## Configuration

A symbolic link has been created for easier Kubernetes configuration:

```
~/.kube/config → /home/ysha/codebase/k8s-demo/setup/kubeconfig.yaml
```

## Benefits

1. **Default Path Works**: All examples using `~/.kube/config` now work automatically
2. **kubectl Works**: Standard `kubectl` commands work without specifying `KUBECONFIG`
3. **Minimal Config**: Can use `00-minimal.json` without specifying kubeconfig path

## Verification

```bash
# Check the symlink
ls -lah ~/.kube/config

# Test kubectl
kubectl get nodes

# Test with minimal config
madengine-cli build --tags dummy --registry dockerhub \
  --additional-context-file examples/k8s-configs/00-minimal.json
```

## How It Was Created

```bash
mkdir -p ~/.kube
ln -s /home/ysha/codebase/k8s-demo/setup/kubeconfig.yaml ~/.kube/config
```

## Updating the Target

If you need to point to a different kubeconfig:

```bash
# Remove old symlink
rm ~/.kube/config

# Create new symlink
ln -s /path/to/new/kubeconfig.yaml ~/.kube/config
```

## Cleanup

If you need to remove the symlink:

```bash
rm ~/.kube/config
```

---

**Created**: December 1, 2025  
**Status**: Active ✅
