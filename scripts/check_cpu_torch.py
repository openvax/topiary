"""Fail CPU CI setup if PyTorch pulled CUDA/cuDNN or Triton wheels (#564)."""

from importlib.metadata import distributions


CUDA_WHEEL_PREFIXES = ("nvidia-",)
CUDA_WHEEL_NAMES = frozenset({"triton"})


def unexpected_cuda_distributions():
    """Return sorted installed distribution names that indicate CUDA wheel bloat."""
    found = set()
    for dist in distributions():
        name = (dist.metadata.get("Name") or "").strip().lower()
        if not name:
            continue
        if name in CUDA_WHEEL_NAMES or name.startswith(CUDA_WHEEL_PREFIXES):
            found.add(name)
    return sorted(found)


def main():
    import torch

    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    if cuda_version is not None:
        raise RuntimeError(
            f"Expected a CPU-only torch wheel on CPU CI, got torch {torch.__version__} "
            f"with CUDA {cuda_version}"
        )
    cuda_packages = unexpected_cuda_distributions()
    if cuda_packages:
        raise RuntimeError(
            "Unexpected CUDA/Triton packages installed on CPU CI: "
            + ", ".join(cuda_packages)
        )
    print(f"Verified CPU torch {torch.__version__} (no CUDA/Triton wheels installed)")


if __name__ == "__main__":
    main()
