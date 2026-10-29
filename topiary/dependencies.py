"""Lazy loading and installation diagnostics for core integrations."""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, requires, version

from packaging.requirements import Requirement


def require_dependency(
    module_name,
    *,
    feature,
    required_callables=(),
):
    """Import a core integration and report actionable installation failures.

    Package-version floors live in one place, Topiary's
    installer metadata, and are enforced here as well as by the installer:
    an older release that imports but gives wrong answers is refused rather
    than trusted. Beyond the floor, this checks what Topiary can answer
    reliably — whether the module and the exact API a feature needs load.
    """
    dependency_name = module_name.partition(".")[0]
    install = "pip install topiary"
    upgrade = "pip install --upgrade topiary"

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

    check_declared_version(dependency_name, feature=feature)

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


def check_declared_version(dependency_name, *, feature):
    """Refuse an installed release outside Topiary's declared range for it.

    Parameters
    ----------
    dependency_name : str
        Distribution name, e.g. ``"isovar"``.
    feature : str
        What needs the dependency, for the error message.

    Raises
    ------
    ImportError
        The installed version does not satisfy the declared specifier.

    Notes
    -----
    Nothing is checked when there is nothing trustworthy to compare: the
    Topiary metadata found describes a different copy of Topiary than the
    one running (a bare source tree, or an older release installed beside
    a checkout, whose floor may be stale), no range is declared, or
    the module is importable without distribution metadata.
    """
    from topiary import __version__ as running

    try:
        if version("topiary") != running:
            return
        declared = requires("topiary") or ()
    except PackageNotFoundError:
        return
    specifiers = [
        requirement.specifier for requirement in map(Requirement, declared)
        if requirement.name == dependency_name
        and (requirement.marker is None or requirement.marker.evaluate({"extra": ""}))
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
            "pip install --upgrade topiary"
        )
