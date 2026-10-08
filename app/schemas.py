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
    # Wan 2.2 Animate: both modes replay a driving video's motion onto a new
    # character. Mix keeps the driving video's background/scene and swaps only
    # the character; Move discards the driving video's background entirely and
    # keeps only its motion, placing the character in a newly generated scene.
    ANIMATE_MIX = "animate_mix"
    ANIMATE_MOVE = "animate_move"
    # Wan 2.2 VACE (Fun variant): several keyframe images placed at chosen frame
    # positions in one clip, not just a start/end pair. The app builds the actual
    # control_video/control_masks fed to WanVaceToVideo from Shot.keyframes.
    VACE = "vace"
    # Jump cut: a montage shot made of still images (generated with /v1/images, uploaded, or read from the
    # inputs directory) cut together by FFmpeg on the server. No GPU, no ComfyUI workflow: the options
    # (per-image duration ratios, hard cut / shake / shutter-wipe transitions, constant camera shake) are in
    # JumpCutSpec. The shot lasts frames / fps seconds like any other shot, so it concatenates with Wan shots.
    JUMPCUT = "jumpcut"


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


class InputVideo(BaseModel):
    """A driving video reference for Wan 2.2 Animate (Mix/Move). Same controlled-source
    contract as InputImage: no arbitrary inline data, only a URL, a path relative to the
    mounted input directory, a (future) uploaded video_id, or the output of an earlier
    completed job (job_id -- used to chain Animate passes, e.g. the two-pass cat-vs-cat
    fight where the second pass replays the first pass's result)."""

    model_config = ConfigDict(extra="forbid")
    url: HttpUrl | None = None
    path: str | None = None
    video_id: str | None = Field(default=None, pattern=r"^vid_[a-f0-9]{32}$")
    job_id: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_-]{1,64}$")

    @model_validator(mode="after")
    def exactly_one_source(self):
        if sum(value is not None for value in (self.url, self.path, self.video_id, self.job_id)) != 1:
            raise ValueError("provide exactly one of url, path, video_id, or job_id")
        if self.path and (self.path.startswith("/") or ".." in self.path.replace("\\", "/").split("/")):
            raise ValueError("path must be relative to the mounted input directory")
        return self


class TimedKeyframe(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image: InputImage
    frame: int = Field(ge=0, description="Frame index (0-based) this image is placed at.")


_NO_NEGATIVE_PROMPT_ENGINES = {"flux_schnell", "krea2"}
_MAX_STEPS_PER_ENGINE = {"flux_schnell": 4, "krea2": 8}


class KeyframeStage(BaseModel):
    engine: Literal["flux_schnell", "krea2", "sd15", "illustrious", "pony", "anima"]
    prompt: str
    negative_prompt: str = ""
    loras: list[LoraUse] = Field(default_factory=list, max_length=8)
    seed: int | None = None
    # flux_schnell: confirmed 4-step distilled default. krea2 (turbo variant):
    # confirmed 8-step default from Comfy-Org's own official workflow template
    # (image_krea2_turbo_t2i.json, fetched 2026-09-28) -- not a guess. The bound
    # is the max of both; _MAX_STEPS_PER_ENGINE enforces each engine's own cap.
    steps: int = Field(default=4, ge=1, le=8)
    guidance: float = Field(default=3.5, ge=0.0, le=100.0)

    @model_validator(mode="after")
    def flux_parameters(self):
        if self.engine in _NO_NEGATIVE_PROMPT_ENGINES and self.negative_prompt.strip():
            raise ValueError(f"{self.engine} does not support a negative prompt")
        if self.engine == "krea2" and "steps" not in self.model_fields_set:
            self.steps = 8  # krea2's own confirmed default, not flux_schnell's 4
        limit = _MAX_STEPS_PER_ENGINE.get(self.engine)
        if limit is not None and self.steps > limit:
            raise ValueError(f"{self.engine} supports at most {limit} steps")
        return self


class ImageGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    engine: Literal["flux_schnell", "krea2"] = "flux_schnell"
    prompt: str = Field(min_length=1, max_length=4000)
    negative_prompt: str = ""
    width: int = Field(default=832, ge=256, le=1536, multiple_of=16)
    height: int = Field(default=480, ge=256, le=1536, multiple_of=16)
    seed: int | None = None
    steps: int = Field(default=4, ge=1, le=8)
    guidance: float = Field(default=3.5, ge=0.0, le=100.0)
    loras: list[LoraUse] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def native_flux_contract(self):
        if self.engine in _NO_NEGATIVE_PROMPT_ENGINES and self.negative_prompt.strip():
            raise ValueError(f"{self.engine} does not support a negative prompt")
        if any(item.target not in {"auto", "keyframe"} for item in self.loras):
            raise ValueError(f"{self.engine} image LoRA target must be auto or keyframe")
        if self.engine == "krea2" and "steps" not in self.model_fields_set:
            self.steps = 8
        limit = _MAX_STEPS_PER_ENGINE.get(self.engine)
        if limit is not None and self.steps > limit:
            raise ValueError(f"{self.engine} supports at most {limit} steps")
        return self


# Mirrors scripts/trend_philosopher.py (BURST_TRANSITIONS, CONSTANT_SHAKES); tests/test_jumpcut_shot.py checks they stay in step.
JumpCutTransition = Literal["cut", "shake", "wipe_left", "wipe_right", "swing"]
JumpCutShake = Literal["jitter", "roll", "handheld", "sway", "rock", "drift"]
JUMPCUT_MIN_IMAGE_SECONDS = 0.05


class JumpCutSpec(BaseModel):
    """What a jump cut shot shows and how it is cut. Every field has a default, so {"images": [...]} is a complete request."""

    model_config = ConfigDict(extra="forbid")
    images: list[InputImage] = Field(min_length=2, max_length=40, description="The stills, in the order they are shown.")
    # One number per image: image i stays ratios[i] / sum(ratios) of the shot ([1, 1, 2, 1] on 5 s = 1, 1, 2, 1 s). More ratios than
    # images cycle through the images. None = automatic: shuffled images, each shown cut_min_seconds..cut_max_seconds.
    ratios: list[float] | None = Field(default=None, min_length=1, max_length=200)
    # How one image gives way to the next: "cut" hard cut, "shake" hard cut + a jolt, "wipe_left"/"wipe_right" shutter wipe, "swing" the wipe alternates.
    transition: JumpCutTransition = "cut"
    transition_seconds: float = Field(default=0.10, gt=0.0, le=1.0)
    # A constant camera shake on top of the transitions, for the whole shot ("rock" = slow, smooth, framing unchanged).
    constant_shake: JumpCutShake | None = None
    constant_shake_amount: float = Field(default=1.0, gt=0.0, le=3.0)
    cut_min_seconds: float = Field(default=0.2, ge=JUMPCUT_MIN_IMAGE_SECONDS, le=5.0)
    cut_max_seconds: float = Field(default=0.4, ge=JUMPCUT_MIN_IMAGE_SECONDS, le=5.0)

    @model_validator(mode="after")
    def coherent(self):
        if self.ratios is not None and any(value <= 0 for value in self.ratios):
            raise ValueError("ratios must all be positive")
        if self.cut_min_seconds > self.cut_max_seconds:
            raise ValueError("cut_min_seconds must not exceed cut_max_seconds")
        return self


class Shot(BaseModel):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    mode: Mode
    prompt: str = ""
    negative_prompt: str = ""
    start_image: InputImage | None = None
    end_image: InputImage | None = None
    generate_start_image: KeyframeStage | None = None
    driving_video: InputVideo | None = None
    # How a non-square driving clip becomes the square the Animate template
    # works on: "crop" is the template's own center crop, "pad" letterboxes so
    # action at the sides of a wide clip is kept (see app/driving_video.py).
    driving_fit: Literal["crop", "pad"] = "crop"
    # animate_mix only: (x, y) in 0..1 of the driving clip's first frame, on the
    # person to replace. Overrides the automatic YOLO "best person" pick, which is
    # wrong for clips with two people. The subject must be visible in frame 0.
    # Where the skeleton that drives the motion comes from. "human" (default) is the
    # validated DWPose body+hands+face pipeline; "animal" swaps in the AP10K quadruped
    # skeleton of comfyui_controlnet_aux for a pet as the motion source. Experimental:
    # nothing documents Wan 2.2 Animate as trained on that skeleton, and a character
    # sitting upright like a person does not match a quadruped skeleton -- use a human
    # performer for an upright/anthropomorphic character.
    pose_source: Literal["human", "animal"] = "human"
    subject_point: tuple[float, float] | None = None
    # animate_mix only: red SAM2 points, (x, y) in 0..1 of the same first frame, on what
    # must NOT be segmented (background, the other fighter). subject_point is the green
    # point; both kinds are PointsEditor points, and only green is wired in the stock
    # template -- this connects the red ones. Not live-validated yet (2026-10-02).
    exclude_points: list[tuple[float, float]] = Field(default_factory=list, max_length=8)
    keyframes: list[TimedKeyframe] = Field(default_factory=list, max_length=32)
    width: int = Field(default=832, ge=256, le=1536, multiple_of=16)
    height: int = Field(default=480, ge=256, le=1536, multiple_of=16)
    frames: int = Field(default=121, ge=17, le=241)
    fps: int = Field(default=24, ge=1, le=60)
    seed: int | None = None
    steps: int = Field(default=20, ge=1, le=80)
    cfg: float = Field(default=5.0, ge=0.0, le=20.0)
    turbo_mode: bool = False
    loras: list[LoraUse] = Field(default_factory=list, max_length=8)
    # jumpcut shots only (see Mode.JUMPCUT)
    jumpcut: JumpCutSpec | None = None

    @model_validator(mode="after")
    def mode_inputs(self):
        if self.mode == Mode.JUMPCUT:
            _check_jumpcut_shot(self)
            return self
        if self.jumpcut is not None:
            raise ValueError("jumpcut is only accepted for jumpcut shots")
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
            Mode.ANIMATE_MIX,
            Mode.ANIMATE_MOVE,
        }
        if needs_start and self.start_image is None and self.generate_start_image is None:
            raise ValueError(f"{self.mode.value} requires start_image or generate_start_image")
        if self.start_image is not None and self.generate_start_image is not None:
            raise ValueError("start_image and generate_start_image are mutually exclusive")
        if self.mode in {Mode.TEXT_KEYFRAMES_TO_VIDEO, Mode.KEYFRAMES_TO_VIDEO} and self.end_image is None:
            raise ValueError(f"{self.mode.value} requires end_image")
        if self.mode in {Mode.T2V, Mode.TEXT_IMAGE_TO_VIDEO, Mode.TEXT_KEYFRAMES_TO_VIDEO} and not self.prompt.strip():
            raise ValueError(f"{self.mode.value} requires a prompt")
        if self.mode in {Mode.ANIMATE_MIX, Mode.ANIMATE_MOVE} and self.driving_video is None:
            raise ValueError(f"{self.mode.value} requires driving_video (the performer video to replay)")
        if self.driving_video is not None and self.mode not in {Mode.ANIMATE_MIX, Mode.ANIMATE_MOVE}:
            raise ValueError("driving_video is only accepted for animate_mix/animate_move shots")
        if self.pose_source != "human" and self.mode not in {Mode.ANIMATE_MIX, Mode.ANIMATE_MOVE}:
            raise ValueError("pose_source is only accepted for animate_mix/animate_move shots")
        if self.subject_point is not None or self.exclude_points:
            if self.mode != Mode.ANIMATE_MIX:
                raise ValueError(
                    "subject_point/exclude_points are only accepted for animate_mix shots (move mode has no SAM2 mask)"
                )
            for point in ([self.subject_point] if self.subject_point is not None else []) + list(self.exclude_points):
                if not all(0.0 <= value <= 1.0 for value in point):
                    raise ValueError("subject_point/exclude_points must be (x, y) with both values between 0 and 1")
        if self.mode in {Mode.ANIMATE_MIX, Mode.ANIMATE_MOVE}:
            # The template squares the driving video (640x640) and then crops it
            # again to width x height, so any non-square job silently loses the top
            # and bottom (or sides) of the pose and mask. Square is the only size
            # that renders the whole performer; 512 is the validated default.
            # The model ships distilled (LightX2V): the generic 20 steps / cfg 5 would be slow and
            # over-baked. Every live run (2026-10-02/03) used 8 steps, cfg 1.
            if "steps" not in self.model_fields_set:
                self.steps = 8
            if "cfg" not in self.model_fields_set:
                self.cfg = 1.0
            width_set, height_set = "width" in self.model_fields_set, "height" in self.model_fields_set
            if not width_set and not height_set:
                self.width = self.height = 512
            elif width_set and not height_set:
                self.height = self.width
            elif height_set and not width_set:
                self.width = self.height
            elif self.width != self.height:
                raise ValueError(
                    "animate shots must be square (width == height): the template center-crops "
                    f"the driving video to a square, got {self.width}x{self.height}"
                )
        if self.mode == Mode.VACE:
            if not self.prompt.strip():
                raise ValueError(f"{self.mode.value} requires a prompt")
            if len(self.keyframes) < 2:
                raise ValueError(f"{self.mode.value} requires at least 2 keyframes")
            frame_indices = [item.frame for item in self.keyframes]
            if len(set(frame_indices)) != len(frame_indices):
                raise ValueError("keyframes must have distinct frame indices")
            if max(frame_indices) >= self.frames:
                raise ValueError(
                    f"keyframe frame index {max(frame_indices)} is out of range for {self.frames} frames"
                )
        elif self.keyframes:
            raise ValueError("keyframes is only accepted for vace shots")
        # Note: turbo_mode is already rejected for these modes by the earlier
        # "turbo_mode is supported only for T2V/I2V workflows" check above.
        return self


def _check_jumpcut_shot(shot: "Shot") -> None:
    if shot.jumpcut is None:
        raise ValueError("jumpcut requires the jumpcut options (at least its images)")
    gpu_inputs = (
        shot.start_image, shot.end_image, shot.generate_start_image, shot.driving_video, shot.keyframes or None, shot.loras or None,
    )
    if any(item is not None for item in gpu_inputs) or shot.turbo_mode:
        raise ValueError(
            "a jumpcut shot is rendered by FFmpeg from jumpcut.images: start_image, end_image, generate_start_image, "
            "driving_video, keyframes, loras and turbo_mode are not accepted"
        )
    if shot.jumpcut.ratios:
        shortest = shot.frames / shot.fps * min(shot.jumpcut.ratios) / sum(shot.jumpcut.ratios)
        if shortest < JUMPCUT_MIN_IMAGE_SECONDS:
            raise ValueError(
                f"with these ratios an image would stay only {shortest:.3f} s: use fewer ratios, a longer shot (frames), or less extreme ratios"
            )


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
