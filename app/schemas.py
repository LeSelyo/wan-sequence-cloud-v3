from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class Mode(str, Enum):
    T2V = "t2v"
    TEXT_IMAGE_TO_VIDEO = "t+i2v"
    I2V = "i2v"
    TEXT_KEYFRAMES_TO_VIDEO = "t+i(keyframe)2v"
    KEYFRAMES_TO_VIDEO = "i(keyframe)2v"


class LoraUse(BaseModel):
    id: str
    weight: float = Field(default=1.0, ge=-2.0, le=2.0)
    target: Literal["auto", "high", "low", "both", "keyframe"] = "auto"


class InputImage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: HttpUrl | None = None
    path: str | None = None
    image_id: str | None = Field(default=None, pattern=r"^img_[a-f0-9]{32}$")

    @model_validator(mode="after")
    def exactly_one_source(self):
        if sum(value is not None for value in (self.url, self.path, self.image_id)) != 1:
            raise ValueError("provide exactly one of url, path, or image_id")
        if self.path and (self.path.startswith("/") or ".." in self.path.replace("\\", "/").split("/")):
            raise ValueError("path must be relative to the mounted input directory")
        return self


class KeyframeStage(BaseModel):
    engine: Literal["flux_schnell", "sd15", "illustrious", "pony", "anima"]
    prompt: str
    negative_prompt: str = ""
    loras: list[LoraUse] = Field(default_factory=list, max_length=8)
    seed: int | None = None
    steps: int = Field(default=4, ge=1, le=4)
    guidance: float = Field(default=3.5, ge=0.0, le=100.0)

    @model_validator(mode="after")
    def flux_parameters(self):
        if self.engine == "flux_schnell" and self.negative_prompt.strip():
            raise ValueError("flux_schnell does not support a negative prompt")
        return self


class ImageGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    engine: Literal["flux_schnell"] = "flux_schnell"
    prompt: str = Field(min_length=1, max_length=4000)
    negative_prompt: str = ""
    width: int = Field(default=832, ge=256, le=1536, multiple_of=16)
    height: int = Field(default=480, ge=256, le=1536, multiple_of=16)
    seed: int | None = None
    steps: int = Field(default=4, ge=1, le=4)
    guidance: float = Field(default=3.5, ge=0.0, le=100.0)
    loras: list[LoraUse] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def native_flux_contract(self):
        if self.negative_prompt.strip():
            raise ValueError("flux_schnell does not support a negative prompt")
        if any(item.target not in {"auto", "keyframe"} for item in self.loras):
            raise ValueError("flux_schnell image LoRA target must be auto or keyframe")
        return self


class Shot(BaseModel):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    mode: Mode
    prompt: str = ""
    negative_prompt: str = ""
    start_image: InputImage | None = None
    end_image: InputImage | None = None
    generate_start_image: KeyframeStage | None = None
    width: int = Field(default=832, ge=256, le=1536, multiple_of=16)
    height: int = Field(default=480, ge=256, le=1536, multiple_of=16)
    frames: int = Field(default=121, ge=17, le=241)
    fps: int = Field(default=24, ge=1, le=60)
    seed: int | None = None
    steps: int = Field(default=20, ge=1, le=80)
    cfg: float = Field(default=5.0, ge=0.0, le=20.0)
    turbo_mode: bool = False
    loras: list[LoraUse] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def mode_inputs(self):
        if self.turbo_mode:
            if self.mode not in {Mode.T2V, Mode.TEXT_IMAGE_TO_VIDEO, Mode.I2V}:
                raise ValueError("turbo_mode is supported only for T2V/I2V workflows")
            if "steps" in self.model_fields_set and self.steps != 4:
                raise ValueError("turbo_mode requires steps=4 from the pinned workflow")
            if "cfg" in self.model_fields_set and self.cfg != 1.0:
                raise ValueError("turbo_mode requires cfg=1.0 from the pinned workflow")
            self.steps = 4
            self.cfg = 1.0
        needs_start = self.mode in {
            Mode.TEXT_IMAGE_TO_VIDEO,
            Mode.I2V,
            Mode.TEXT_KEYFRAMES_TO_VIDEO,
            Mode.KEYFRAMES_TO_VIDEO,
        }
        if needs_start and self.start_image is None and self.generate_start_image is None:
            raise ValueError(f"{self.mode.value} requires start_image or generate_start_image")
        if self.start_image is not None and self.generate_start_image is not None:
            raise ValueError("start_image and generate_start_image are mutually exclusive")
        if self.mode in {Mode.TEXT_KEYFRAMES_TO_VIDEO, Mode.KEYFRAMES_TO_VIDEO} and self.end_image is None:
            raise ValueError(f"{self.mode.value} requires end_image")
        if self.mode in {Mode.T2V, Mode.TEXT_IMAGE_TO_VIDEO, Mode.TEXT_KEYFRAMES_TO_VIDEO} and not self.prompt.strip():
            raise ValueError(f"{self.mode.value} requires a prompt")
        return self


class Transition(BaseModel):
    type: Literal["cut", "crossfade"] = "cut"
    duration_seconds: float = Field(default=0.25, ge=0.0, le=3.0)


class SequenceRequest(BaseModel):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    shots: list[Shot] = Field(min_length=1, max_length=100)
    transition: Transition = Field(default_factory=Transition)
    output_format: Literal["mp4"] = "mp4"


class JobAccepted(BaseModel):
    job_id: str
    status: str
    status_url: str
    estimated_cost_units: float


class ImageJobAccepted(BaseModel):
    image_id: str
    status: str
    status_url: str
    output_url: str
