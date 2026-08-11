from __future__ import annotations

import os
import shutil
import tempfile
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path

from fastapi import UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

from .settings import Settings


ALLOWED_IMAGE_FORMATS = {"PNG", "JPEG"}
STREAM_BLOCK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class StoredUpload:
    image_id: str
    path: Path
    original_format: str
    width: int
    height: int


class ImageUploadError(ValueError):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


async def store_uploaded_image(upload: UploadFile, settings: Settings) -> StoredUpload:
    limit_bytes = settings.max_input_image_mb * 1024 * 1024
    image_root = settings.image_outputs_dir.resolve()
    image_root.mkdir(parents=True, exist_ok=True)
    input_fd, input_name = tempfile.mkstemp(
        prefix=".upload-", suffix=".part", dir=image_root
    )
    os.close(input_fd)
    input_temp = Path(input_name)
    canonical_temp = input_temp.with_suffix(".png.part")
    output_dir: Path | None = None
    try:
        received = 0
        with input_temp.open("wb") as handle:
            while chunk := await upload.read(STREAM_BLOCK_BYTES):
                received += len(chunk)
                if received > limit_bytes:
                    raise ImageUploadError(
                        413,
                        f"uploaded image exceeds {settings.max_input_image_mb} MiB limit",
                    )
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        if received == 0:
            raise ImageUploadError(422, "uploaded image is empty")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(input_temp) as source:
                    original_format = str(source.format or "").upper()
                    if original_format not in ALLOWED_IMAGE_FORMATS:
                        raise ImageUploadError(
                            415, "uploaded image must decode as PNG or JPEG"
                        )
                    width, height = source.size
                    if width <= 0 or height <= 0:
                        raise ImageUploadError(422, "uploaded image has invalid dimensions")
                    if width * height > settings.max_input_image_pixels:
                        raise ImageUploadError(
                            413,
                            "uploaded image exceeds configured pixel limit",
                        )
                    if int(getattr(source, "n_frames", 1)) != 1:
                        raise ImageUploadError(422, "animated images are not supported")
                    canonical = ImageOps.exif_transpose(source)
                    canonical.load()
                    width, height = canonical.size
                    if canonical.mode not in {"RGB", "RGBA"}:
                        canonical = canonical.convert("RGBA")
                    with canonical_temp.open("wb") as output:
                        canonical.save(output, format="PNG", optimize=False)
                        output.flush()
                        os.fsync(output.fileno())
        except ImageUploadError:
            raise
        except (Image.DecompressionBombError, Image.DecompressionBombWarning):
            raise ImageUploadError(413, "uploaded image is a decompression bomb") from None
        except (UnidentifiedImageError, OSError, ValueError):
            raise ImageUploadError(422, "uploaded file is not a valid image") from None

        image_id = f"img_{uuid.uuid4().hex}"
        output_dir = image_root / image_id
        output_dir.mkdir(mode=0o775)
        target = output_dir / "image.png"
        os.replace(canonical_temp, target)
        return StoredUpload(image_id, target, original_format, width, height)
    except Exception:
        if output_dir is not None and output_dir.parent == image_root:
            shutil.rmtree(output_dir, ignore_errors=True)
        raise
    finally:
        input_temp.unlink(missing_ok=True)
        canonical_temp.unlink(missing_ok=True)
        await upload.close()
