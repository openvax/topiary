"""Shared lazy loading for Topiary's optional integrations."""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, requires, version

from packaging.requirements import Requirement


def require_optional_dependency(
    module_name,
    *,
    feature,
    extra=None,
    required_callables=(),
):
    """Import an optional module and report actionable failures.

    Package-version floors live in one place, the extra in Topiary's
    installer metadata, and are enforced here as well as by the installer:
    an older release that imports but gives wrong answers is refused rather
    than trusted. Beyond the floor, this checks what Topiary can answer
    reliably — whether the module and the exact API a feature needs load.
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

    check_declared_version(dependency_name, extra=extra, feature=feature)

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


def check_declared_version(dependency_name, *, extra=None, feature):
    """Refuse an installed release outside Topiary's declared range for it.

    Parameters
    ----------
    dependency_name : str
        Distribution name, e.g. ``"isovar"``.
    extra : str, optional
        The Topiary extra that declares it; defaults to the same name.
    feature : str
        What needs the dependency, for the error message.

    Raises
    ------
    ImportError
        The installed version does not satisfy the extra's specifier.

    Notes
    -----
    Nothing is checked when there is nothing to compare: Topiary itself is
    not installed (a bare source tree has no metadata), the extra declares
    no range, or the module is importable without distribution metadata.
    """
    extra = dependency_name if extra is None else extra
    try:
        declared = requires("topiary") or ()
    except PackageNotFoundError:
        return
    specifiers = [
        requirement.specifier for requirement in map(Requirement, declared)
        if requirement.name == dependency_name and requirement.marker is not None
        and requirement.marker.evaluate({"extra": extra})
    ]
    if not specifiers or not str(specifiers[0]):
        return
    try:
        installed = version(dependency_name)
    except PackageNotFoundError:
        return
    if not specifiers[0].contains(installed, prereleases=True):
        raise ImportError(
            f"{dependency_name} {installed} is installed, but {feature} requires "
            f"{dependency_name}{specifiers[0]}. Upgrade with: "
            f"pip install --upgrade 'topiary[{extra}]'"
        )
