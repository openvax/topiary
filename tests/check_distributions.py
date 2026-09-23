"""Check built artifacts without importing Topiary or installing its dependencies.

Run after ``python -m build`` with ``python tests/check_distributions.py dist``.
The normal build creates the wheel from the sdist, so this checks both sides
of the release pipeline. Only standard-library modules are needed.
"""

import argparse
import configparser
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path
import tarfile
import zipfile


def check_license_metadata(metadata):
    """Require PEP 639 metadata without a deprecated license classifier."""
    assert metadata["License-Expression"] == "Apache-2.0"
    assert metadata.get_all("License-File") == ["LICENSE"]
    assert not any(
        value.startswith("License ::")
        for value in metadata.get_all("Classifier", [])
    )


def check_runtime_metadata(metadata):
    """Ordinary installation must provide Osteosarc reads on supported Python."""
    assert metadata["Requires-Python"] == ">=3.10"
    requirements = [value.replace(" ", "") for value in metadata.get_all("Requires-Dist", [])]
    assert any(value in ("osteosarc>=0.2.3,<0.3", "osteosarc<0.3,>=0.2.3")
               for value in requirements)
    assert "osteosarc" not in metadata.get_all("Provides-Extra", [])


def check_distributions(dist_dir, source_root):
    """Verify runtime files, source resources, license text, and the CLI entry."""
    wheels = list(dist_dir.glob("*.whl"))
    sdists = list(dist_dir.glob("*.tar.gz"))
    assert len(wheels) == 1, f"Expected one wheel in {dist_dir}, found {wheels}"
    assert len(sdists) == 1, f"Expected one sdist in {dist_dir}, found {sdists}"
    runtime_files = {
        path.relative_to(source_root).as_posix()
        for path in (source_root / "topiary").rglob("*.py")
    }
    license_text = (source_root / "LICENSE").read_bytes()

    with zipfile.ZipFile(wheels[0]) as wheel:
        names = set(wheel.namelist())
        metadata_paths = [name for name in names if name.endswith(".dist-info/METADATA")]
        assert len(metadata_paths) == 1
        dist_info = metadata_paths[0].split("/")[0]
        unexpected = {
            name for name in names
            if name.split("/")[0] not in {"topiary", dist_info}
        }
        assert not unexpected, f"Non-runtime files in wheel: {sorted(unexpected)}"
        assert runtime_files <= names, f"Missing runtime modules: {runtime_files - names}"
        assert wheel.read(f"{dist_info}/top_level.txt").decode().splitlines() == ["topiary"]
        assert wheel.read(f"{dist_info}/licenses/LICENSE") == license_text
        wheel_metadata = BytesParser().parsebytes(wheel.read(metadata_paths[0]))
        check_license_metadata(wheel_metadata)
        check_runtime_metadata(wheel_metadata)
        entry_points = configparser.ConfigParser()
        entry_points.read_string(wheel.read(f"{dist_info}/entry_points.txt").decode())
        assert entry_points["console_scripts"]["topiary"] == "topiary.cli.script:main"

    with tarfile.open(sdists[0]) as sdist:
        files = {member.name: member for member in sdist.getmembers() if member.isfile()}
        roots = {name.split("/")[0] for name in files}
        assert len(roots) == 1
        root = roots.pop()
        relative_names = {name[len(root) + 1:] for name in files}
        source_resources = {
            path.relative_to(source_root).as_posix()
            for directory, pattern in (
                ("docs", "*.md"), ("tests", "*.py"), ("tests/data", "*"),
                ("scripts", "*.sh"), ("scripts", "*.py"), (".github/workflows", "*.yml"),
            )
            for path in (source_root / directory).rglob(pattern)
            if path.is_file() and "__pycache__" not in path.parts
        }
        required = runtime_files | source_resources | {
            "README.md", "CHANGELOG.md", "LICENSE", "pyproject.toml",
            "requirements.txt", "mkdocs.yml", "lint.sh", "test.sh", "deploy.sh",
        }
        assert required <= relative_names, f"Missing source files: {required - relative_names}"
        manifest_path = f"{root}/tests/data/manifest.json"
        manifest = json.loads(sdist.extractfile(files[manifest_path]).read())
        total = 0
        expected_reads = set()
        for asset in manifest["assets"]:
            name = "tests/data/" + asset["filename"]
            contents = sdist.extractfile(files[f"{root}/{name}"]).read()
            assert len(contents) == asset["size_bytes"], name
            assert hashlib.sha256(contents).hexdigest() == asset["sha256"], name
            total += len(contents)
            if name.endswith((".bam", ".bam.bai", ".sam.gz")):
                expected_reads.add(name)
        assert total == manifest["total_size_bytes"]
        assert total <= 11 * 1024 * 1024, "Review Sid fixture growth before increasing the 11 MiB budget"
        bundled_reads = {name for name in relative_names
                         if any(name.startswith("tests/data/" + group + "/") for group in manifest["groups"])
                         and name.endswith((".bam", ".bam.bai", ".sam.gz"))}
        assert bundled_reads == expected_reads, "Unselected Sid reads entered the source distribution"
        assert sdist.extractfile(files[f"{root}/LICENSE"]).read() == license_text
        source_metadata = BytesParser().parsebytes(
            sdist.extractfile(files[f"{root}/PKG-INFO"]).read()
        )
        check_license_metadata(source_metadata)
        check_runtime_metadata(source_metadata)
        for field in ("Name", "Version", "Requires-Python", "Requires-Dist", "Provides-Extra"):
            assert source_metadata.get_all(field) == wheel_metadata.get_all(field), field

    print(f"Validated {wheels[0].name} and {sdists[0].name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist_dir", type=Path)
    args = parser.parse_args()
    check_distributions(args.dist_dir, Path(__file__).resolve().parents[1])
