from pathlib import Path
from zipfile import ZipFile

import pytest

from image_generator.engine import create_outputs_zip, sanitize_prompt


def test_sanitize_prompt_trims_whitespace():
    assert sanitize_prompt("  hello world  ") == "hello world"


def test_sanitize_prompt_rejects_empty_values():
    with pytest.raises(ValueError):
        sanitize_prompt("   ")


def test_create_outputs_zip_creates_archive(tmp_path: Path):
    image_1 = tmp_path / "a.png"
    image_2 = tmp_path / "b.png"
    image_1.write_bytes(b"png-a")
    image_2.write_bytes(b"png-b")

    zip_path = create_outputs_zip(tmp_path)

    assert zip_path.exists()
    with ZipFile(zip_path) as archive:
        assert archive.namelist() == ["a.png", "b.png"]


def test_create_outputs_zip_raises_when_no_images(tmp_path: Path):
    with pytest.raises(ValueError):
        create_outputs_zip(tmp_path)
