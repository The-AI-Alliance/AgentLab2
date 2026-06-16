#!/usr/bin/env bash
# Install QEMU system emulator for the current platform.
# Called from `make install` — safe to run on Linux and macOS.
set -euo pipefail

case "$(uname)" in
  Linux)
    if ! command -v qemu-system-x86_64 &>/dev/null; then
      sudo apt-get update -qq
      sudo apt-get install -y qemu-system-x86 qemu-utils
    fi
    ;;
  Darwin)
    if ! command -v qemu-system-x86_64 &>/dev/null; then
      brew install qemu
    fi
    ;;
  MINGW*|MSYS*|CYGWIN*)
    echo "Windows is not supported. Run osworld-cube under WSL2 (Linux)." >&2
    exit 1
    ;;
  *)
    echo "Unsupported platform: $(uname). Install qemu-system-x86_64 manually." >&2
    exit 1
    ;;
esac

echo "QEMU installed: $(qemu-system-x86_64 --version | head -1)"

if [ -e /dev/kvm ]; then
  echo "KVM: available — VMs will use hardware acceleration."
else
  echo "KVM: not available — VMs will run under TCG (software emulation, significantly slower)."
fi
