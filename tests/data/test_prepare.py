import json
from pathlib import Path

from alm.data.config import load_corpus_config
from alm.data.prepare import prepare_all


def test_prepare_local_source(tmp_path: Path) -> None:
    source_txt = tmp_path / "sample.txt"
    source_txt.write_text("Hello\nWorld\n")
    config_yaml = tmp_path / "corpus.yaml"
    config_yaml.write_text(
        """
        sources:
          sample:
            kind: local
            path: {path}
        """.format(path=str(source_txt))
    )
    config = load_corpus_config(config_yaml)
    out_dir = tmp_path / "clean"
    prepare_all(config, out_dir)
    cleaned = (out_dir / "sample.txt").read_text()
    metadata = json.loads((out_dir / "sample.json").read_text())
    assert "Hello" in cleaned
    assert metadata["lines"] == 2
