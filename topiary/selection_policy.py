"""Portable selection policy definitions over Topiary's existing DSL."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

from .ranking import as_dsl_node


_POLICY_FIELDS = {"schema_version", "name", "score_by", "filter_by", "ascending",
                  "duplicates", "strata", "default_methods", "default_versions"}


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


@dataclass(frozen=True)
class SelectionPolicy:
    """A named filter and scoring policy for combined source candidates.

    Parameters
    ----------
    name : str
        User-assigned policy identifier, for example ``openvax-v1``. Give a
        changed policy a new name; the content digest also distinguishes
        different definitions carrying the same name.
    score_by : str
        Numeric Topiary DSL expression. There is no implicit scientific
        recipe: callers must state the expression they want to preserve.
    filter_by : str, optional
        Boolean Topiary DSL expression; None applies no additional filter.
        Existing ranker semantics apply, including exclusion on unknown
        filter results and retention of missing scores as unranked candidates.
    ascending : bool
        False ranks larger scores first; True ranks smaller scores first.
    duplicates : {'error', 'best', 'worst'}
        Policy for conflicting observations of a candidate. Default error
        requires an explicit choice instead of silently choosing evidence.
    strata : sequence of str
        Independent ranking partitions, defaulting to MHC class. An empty
        sequence explicitly requests a ranking across classes.
    default_methods : mapping of str to str, optional
        Explicit kind-to-method selections. None leaves model resolution to
        the existing DSL; no model is selected from installed software.
    default_versions : mapping of (str, str) to str, optional
        Explicit (kind, method)-to-version selections. Unknown source versions
        remain unknown. JSON represents these tuple keys as records.

    Notes
    -----
    Expressions are validated when the policy is created. Referenced input
    columns and model selections are resolved by the existing ranker when
    the policy is applied. Sequences and mappings are copied and made immutable,
    so editing the caller's
    configuration cannot change a saved policy. Policies do not run predictors.
    """

    name: str
    score_by: str
    filter_by: str | None = None
    ascending: bool = False
    duplicates: str = "error"
    strata: tuple[str, ...] = ("candidate_mhc_class",)
    default_methods: Mapping | None = None
    default_versions: Mapping | None = None

    def __post_init__(self):
        _text(self.name, "name")
        for label in ("score_by", "filter_by"):
            value = getattr(self, label)
            if value is None and label == "filter_by":
                continue
            try:
                as_dsl_node(_text(value, label))
            except (ValueError, SyntaxError) as error:
                raise ValueError(f"{label}: {error}") from error
        if type(self.ascending) is not bool:
            raise ValueError("ascending must be a boolean")
        if not isinstance(self.duplicates, str) or self.duplicates not in {"error", "best", "worst"}:
            raise ValueError("duplicates must be 'error', 'best', or 'worst'")
        if isinstance(self.strata, str) or not isinstance(self.strata, (list, tuple)):
            raise ValueError("strata must be a sequence of distinct column names")
        strata = tuple(_text(value, "stratum") for value in self.strata)
        if len(set(strata)) != len(strata):
            raise ValueError("strata must name distinct columns")
        object.__setattr__(self, "strata", strata)
        for label in ("default_methods", "default_versions"):
            supplied = getattr(self, label)
            if supplied is None:
                continue
            if not isinstance(supplied, Mapping):
                raise ValueError(f"{label} must be a mapping or None")
            copied = dict(supplied)
            for key, value in copied.items():
                if label == "default_versions":
                    if not isinstance(key, tuple) or len(key) != 2:
                        raise ValueError("default_versions keys must be (kind, method) pairs")
                    for part in key:
                        _text(part, "version selection kind/method")
                else:
                    _text(key, "method selection kind")
                _text(value, label)
            object.__setattr__(self, label, MappingProxyType(copied))

    def to_dict(self):
        """Return an independent, JSON-compatible schema-version-1 definition.

        Explicit None selections remain None; empty mappings remain empty.
        Every setting is written, including defaults, so a future constructor
        default cannot alter the meaning of the saved definition.

        Returns
        -------
        dict
            Complete definition with string keys, suitable for JSON or YAML.
        """
        versions = self.default_versions
        return dict(
            schema_version=1, name=self.name, score_by=self.score_by,
            filter_by=self.filter_by, ascending=self.ascending,
            duplicates=self.duplicates, strata=list(self.strata),
            default_methods=None if self.default_methods is None else dict(sorted(self.default_methods.items())),
            default_versions=None if versions is None else [
                dict(kind=kind, method=method, version=version)
                for (kind, method), version in sorted(versions.items())],
        )

    @classmethod
    def from_dict(cls, definition):
        """Load a complete saved definition, rejecting unknown/missing fields.

        Only schema version 1 is accepted. Duplicate model/version selections
        raise instead of silently taking the last record.

        Parameters
        ----------
        definition : mapping
            Complete output of ``to_dict()``, including explicit defaults.
            Empty, partial or extra-field definitions raise ValueError.

        Returns
        -------
        SelectionPolicy
            Validated immutable policy, independent of the supplied mapping.
        """
        if not isinstance(definition, Mapping):
            raise ValueError("SelectionPolicy definition must be a mapping")
        missing, unknown = _POLICY_FIELDS - set(definition), set(definition) - _POLICY_FIELDS
        if missing or unknown:
            raise ValueError(f"SelectionPolicy fields: missing={sorted(missing)!r}, unknown={sorted(unknown, key=str)!r}")
        if type(definition["schema_version"]) is not int or definition["schema_version"] != 1:
            raise ValueError("Unsupported selection policy schema_version; expected 1")
        values = dict(definition)
        del values["schema_version"]
        versions = values["default_versions"]
        if versions is not None:
            if not isinstance(versions, list):
                raise ValueError("Saved default_versions must be a list of kind/method/version records")
            decoded = {}
            for record in versions:
                if not isinstance(record, Mapping) or set(record) != {"kind", "method", "version"}:
                    raise ValueError("Each version selection must contain kind, method, and version")
                key = (_text(record["kind"], "kind"), _text(record["method"], "method"))
                if key in decoded:
                    raise ValueError(f"Duplicate version selection for {key!r}")
                decoded[key] = record["version"]
            values["default_versions"] = decoded
        return cls(**values)

    @property
    def sha256(self):
        """SHA-256 of the complete canonical definition, including its name."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def resolve_selection_policy(configuration):
    """Resolve an already-composed authoring mapping into a complete policy.

    Parameters
    ----------
    configuration : mapping
        JSON/YAML-compatible settings. ``name`` and ``score_by`` are required;
        other fields use the current constructor defaults if omitted. Version
        selections use the same list of records as :meth:`SelectionPolicy.to_dict`.
        Unknown fields and unsupported schema versions raise ValueError.

    Returns
    -------
    SelectionPolicy
        Immutable effective policy with every default materialized. Persist
        ``to_dict()`` and reload it using ``from_dict()``; do not resolve a
        historical partial configuration against newer defaults.

    Notes
    -----
    Compose authoring overrides *before* this call. An omitted filter can
    inherit a base value during that composition; an explicit null replaces
    it with no filter. This function does not merge files, AND expressions,
    concatenate lists, resolve models from input, or interpret Vaxrank settings.
    Consumers retain their derivation/override records separately from the
    effective definition and digest.
    """
    if not isinstance(configuration, Mapping):
        raise ValueError("SelectionPolicy configuration must be a mapping")
    unknown = set(configuration) - _POLICY_FIELDS
    missing = {"name", "score_by"} - set(configuration)
    if unknown or missing:
        raise ValueError(f"SelectionPolicy fields: missing={sorted(missing)!r}, unknown={sorted(unknown, key=str)!r}")
    defaults = SelectionPolicy(configuration["name"], configuration["score_by"]).to_dict()
    defaults.update(configuration)
    return SelectionPolicy.from_dict(defaults)


def _unique_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate selection policy JSON key: {key!r}")
        result[key] = value
    return result


def read_selection_policy(path):
    """Read a named selection policy from a UTF-8 JSON file.

    The file must contain a complete versioned definition produced by
    :func:`write_selection_policy`. Malformed JSON, duplicate keys, unsupported
    schema versions, and invalid expressions raise ValueError.

    Parameters
    ----------
    path : str or path-like
        Path containing a complete saved definition, not a partial override.

    Returns
    -------
    SelectionPolicy
        Validated, immutable definition. An empty file raises ValueError.
    """
    definition = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_json_keys)
    return SelectionPolicy.from_dict(definition)


def write_selection_policy(policy, path):
    """Save a SelectionPolicy as a new UTF-8 JSON file.

    Existing files raise FileExistsError: save a changed policy under a new
    version/name and path instead of overwriting the historical definition.
    This stores configuration only; observations belong in the result file.

    Parameters
    ----------
    policy : SelectionPolicy
        Complete effective definition to preserve.
    path : str or path-like
        New file to create. Its parent directory must exist.

    Returns
    -------
    None
        The definition is written with all defaults explicitly present.
    """
    if not isinstance(policy, SelectionPolicy):
        raise TypeError("policy must be a SelectionPolicy")
    text = json.dumps(policy.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(text)


def rank_with_policy(result, policy, *, provenance=None):
    """Apply a saved policy to combined source candidates without prediction.

    Parameters
    ----------
    result : TopiaryResult
        Combined source evidence, as accepted by :func:`rank_candidates`.
        It remains unchanged. Re-rank this full evidence to compare policies;
        a previously filtered ranking cannot restore excluded observations.
    policy : SelectionPolicy
        Exact filter, score, selection, and ranking settings to apply.
    provenance : mapping, optional
        JSON-compatible authoring/override provenance, for example ordered
        config file hashes and a base policy digest. It is copied into output
        metadata, outside the effective policy digest. No provenance is inferred
        from a surviving policy name. Unknown provenance remains None.

    Returns
    -------
    TopiaryResult
        The existing ranker's representative rows and candidate scores/ranks,
        with source metadata, the full policy, its SHA-256, and the Topiary
        execution version. Missing scores remain unranked. Empty inputs follow
        the same validation and output rules as :func:`rank_candidates`.
        CSV/TSV exports preserve the definition in ``extra['selection_policy']``.
        This does not add the eligibility/reason view tracked in issue #366.
    """
    from . import __version__
    from .candidates import rank_candidates
    from .result import TopiaryResult
    from .ranking import EvalContext, known_versions

    if not isinstance(policy, SelectionPolicy):
        raise TypeError("policy must be a SelectionPolicy")
    if not isinstance(result, TopiaryResult):
        raise TypeError("result must be a TopiaryResult from combine_sources")
    if provenance is not None and not isinstance(provenance, Mapping):
        raise ValueError("provenance must be a JSON-compatible mapping or None")
    derivation = None if provenance is None else json.loads(json.dumps(dict(provenance), allow_nan=False))
    methods = None if policy.default_methods is None else dict(policy.default_methods)
    versions = None if policy.default_versions is None else dict(policy.default_versions)
    ranked = rank_candidates(
        result, policy.score_by, filter_by=policy.filter_by, ascending=policy.ascending,
        duplicates=policy.duplicates, strata=policy.strata,
        default_methods=methods, default_versions=versions,
    )
    extra = deepcopy(result.extra)
    inventory_columns = [c for c in ("source_label", "kind", "prediction_method_name",
                                     "predictor_version", "prediction_run_name") if c in result.long_df]
    inventory = result.long_df[inventory_columns].drop_duplicates().copy()
    if "predictor_version" in inventory:
        inventory["predictor_version"] = inventory.predictor_version.where(known_versions(inventory.predictor_version))
    inventory = inventory.astype(object).where(inventory.notna(), None)
    extra["selection_policy"] = {
        "definition": policy.to_dict(), "sha256": policy.sha256,
        "provenance": derivation,
        "execution": {
            "topiary_version": __version__, "operation": "rank_candidates",
            "group_keys": list(EvalContext(result.long_df).group_keys),
            "input_topiary_version": result.topiary_version,
            "input_sources": list(result.sources), "input_models": dict(result.models),
            "prediction_inventory": inventory.to_dict(orient="records"),
            "kind_support": deepcopy(result._kind_support()),
        },
    }
    return TopiaryResult(
        ranked, topiary_version=__version__, form="long", sources=list(result.sources),
        models=dict(result.models), filter_by_str=policy.filter_by,
        sort_by_str=policy.score_by, extra=extra,
    )
