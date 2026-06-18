import pytest

from tools.datasets import corpora


class DummyDownloader(corpora.DataDownloader):
    name = "dummy"
    urls = ["dummy.txt"]


@pytest.mark.cpu
def test_tokenize_raises_when_preprocess_command_fails(monkeypatch, tmp_path):
    downloader = DummyDownloader(
        data_dir=str(tmp_path),
        tokenizer_type="CharLevelTokenizer",
        num_workers=1,
    )
    monkeypatch.setattr(corpora.os, "system", lambda cmd: 1)

    with pytest.raises(RuntimeError, match="Failed to tokenize dataset dummy"):
        downloader.tokenize()
