"""Shared lazy loading for Topiary's optional integrations."""

from importlib import import_module


def require_optional_dependency(
    module_name,
    *,
    feature,
    extra=None,
    required_callables=(),
):
    """Import an optional module and report actionable failures.

    Package-version floors belong in installer metadata. Runtime checks here
    cover the question Topiary can answer reliably: whether the module and the
    exact API needed by a feature can be loaded.
    """
    dependency_name = module_name.partition(".")[0]
    extra = dependency_name if extra is None else extra
    install = f"pip install 'topiary[{extra}]'"
    upgrade = f"pip install --upgrade 'topiary[{extra}]'"

    try:
        module = import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == dependency_name:
            raise ImportError(
                f"{dependency_name} is required for {feature}. "
                f"Install it with: {install}"
            ) from error
        raise ImportError(
            f"{dependency_name} is installed but could not be imported for "
            f"{feature}: {error}. Repair its dependencies or reinstall with: "
            f"{upgrade}"
        ) from error
    except Exception as error:
        raise ImportError(
            f"{dependency_name} is installed but could not be imported for "
            f"{feature}: {error}. Repair the installation or reinstall with: "
            f"{upgrade}"
        ) from error

    try:
        missing = [
            name for name in required_callables
            if not callable(getattr(module, name, None))
        ]
    except Exception as error:
        raise ImportError(
            f"{dependency_name} is installed, but loading the API required "
            f"for {feature} failed: {error}. Repair the installation or "
            f"reinstall with: {upgrade}"
        ) from error
    if missing:
        raise ImportError(
            f"{dependency_name} is installed but does not provide the API "
            f"required for {feature}: {', '.join(missing)}. Install a "
            f"compatible release with: {upgrade}"
        )

    return module
