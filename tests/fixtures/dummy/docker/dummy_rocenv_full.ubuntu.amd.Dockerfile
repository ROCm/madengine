# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
ARG BASE_DOCKER=rocm/pytorch
FROM $BASE_DOCKER

# Install system diagnostic tools required by rocenv_tool.py full mode:
#   lshw       -> hardware_information
#   dmidecode  -> bios_settings
#   kmod       -> amdgpu_modinfo (provides modinfo)
#   util-linux -> dmsg_gpu_drm_atom_logs (provides dmesg)
RUN apt-get update && \
    apt-get install -y --no-install-recommends lshw dmidecode kmod util-linux && \
    rm -rf /var/lib/apt/lists/*
