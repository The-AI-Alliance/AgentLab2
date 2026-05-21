#!/usr/bin/env bash
# Install system dependencies for windows-agent-arena-cube.
#
# Required for VM launch (LocalInfraConfig):
#   qemu-system-x86_64 / qemu-img — VM execution and image conversion
#   ovmf                          — UEFI firmware (Windows 11 boots UEFI-only)
#   swtpm                         — TPM 2.0 emulator (required by Windows 11)
#
# Required for image building (packer/run.sh):
#   packer — HashiCorp Packer
#
# Called from `make install` — safe to run on Linux and macOS.
set -euo pipefail

case "$(uname)" in
  Linux)
    needs_install=()
    command -v qemu-system-x86_64 &>/dev/null || needs_install+=(qemu-system-x86 qemu-utils)
    [ -d /usr/share/OVMF ] || needs_install+=(ovmf)
    command -v swtpm &>/dev/null || needs_install+=(swtpm)
    if [ "${#needs_install[@]}" -gt 0 ]; then
      sudo apt-get update -qq
      sudo apt-get install -y "${needs_install[@]}"
    fi
    if ! command -v packer &>/dev/null; then
      curl -fsSL https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg
      echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
      sudo apt-get update -qq
      sudo apt-get install -y packer
    fi
    ;;
  Darwin)
    for pkg in qemu swtpm packer; do
      if ! brew list "$pkg" &>/dev/null; then
        brew install "$pkg"
      fi
    done
    # OVMF on macOS ships inside the qemu formula (share/qemu/edk2-x86_64-code.fd).
    ;;
  MINGW*|MSYS*|CYGWIN*)
    echo "Windows is not supported. Run windows-agent-arena-cube under WSL2 (Linux)." >&2
    exit 1
    ;;
  *)
    echo "Unsupported platform: $(uname). Install qemu-system-x86_64, ovmf, swtpm, packer manually." >&2
    exit 1
    ;;
esac

echo "QEMU:   $(qemu-system-x86_64 --version | head -1)"
echo "swtpm:  $(swtpm --version | head -1)"
echo "packer: $(packer --version | head -1)"

if [ -e /dev/kvm ]; then
  echo "KVM: available — VMs will use hardware acceleration."
else
  echo "KVM: not available — VMs will run under TCG (software emulation, significantly slower)."
fi
