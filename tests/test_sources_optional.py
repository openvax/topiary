import builtins

import pytest

import topiary.sources as sources


def test_check_pirlygenes_raises_clear_error(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pirlygenes" or name.startswith("pirlygenes."):
            raise ImportError("No module named 'pirlygenes'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="pip install pirlygenes"):
        sources._check_pirlygenes()
