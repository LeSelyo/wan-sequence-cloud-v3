# The "you must choose" story video — the automatic procedure

Three lines of context in, a finished vertical (9:16, 720x1280, 30 fps) TikTok video out: second-person apocalypse story, two characters who speak, an A/B choice with a countdown, two branches
(one ends badly, one well), a twist, one-word glitch subtitles, a sound background, a caption with hashtags. **The story is written by a model, not by hand.**

```
python scripts/story_auto.py run the_last_city "A flooded city at night. Two strangers on a rescue boat each offer to save you: a ship captain and a doctor." --seconds 150
python scripts/story_auto.py status the_last_city
```

Everything lands in `results/story_trend/auto/<name>/` (`plan.json`, `cards/`, `run/`, `<name>.mp4`, `caption.txt`, `report.json` with the measured seconds of every step).
Every step is **resumable**: if the box is closed, the PC sleeps or a step fails, run the same command again; a step whose outputs are complete is skipped.

## What the procedure does, step by step

| step | what | where | measured (4090, 150 s video) |
|---|---|---|---|
| `story` | **a chain of agents** (`scripts/story_agents.py`, prompts of the trend in `scripts/trends/you_must_choose.py`): the ANALYST reads the context (three sentences or more) and writes a BRIEF (world, hour, places, the two characters with what they promise and their hidden truth, keywords, facts that must appear); the PLANNER builds the outline act by act (clues, both twists, hook title, closing question, caption), best of N; the WRITER writes the spoken lines chunk by chunk; the DIRECTOR writes the picture and the camera move of every shot; the CODE gives the rhythm (camera never three times alike, shake on impacts), the tags (CASE A / ENDING B), the choice card; a CHECKER validates every answer and tells the model what is wrong (3 attempts); a stage that never succeeds falls back to a template for that stage only | Ollama `qwen3.6:27b` on the box, GPU free | model download 16 GB once; a story is minutes |
| `cards` | the style of the video (art direction read from the context, **the hour of the story too**), a sheet and a portrait per character | Krea2 | ~1 min |
| `closeups` | 5 tight portraits per character, the one whose face is **centred and about 40 % of the frame** is kept (`scripts/face_tools.py`) | Krea2 | ~1.5 min |
| `stills` | one picture per shot that is not a close-up (own prompt, same style prompt) | Krea2, 14 s each | ~8 min for 34 |
| `voices` | one line per shot: a recorded voice is **cloned** (the exact transcript of the sample is made once with Whisper), a character without a matching voice gets one **designed from a text description**; then Whisper gives the **time of every word** | Qwen3-TTS + Whisper | ~5 min for 46 lines |
| `animate` | one **Wan 2.2 S2V** clip per shot: the character speaks his line (portrait + voice), a scene moves (still + silence) — the same model is the image-to-video model | ComfyUI native nodes, LoRA `t2v_1217_low` (4 steps) | ~35-45 s per 3 s clip |
| `qc` | the face of every talking clip is measured (detail / shimmer); a clip that shimmers 35 % more than the median is made again with another seed | PC + box | seconds |
| `post` | **face-detail pass**: ESRGAN x2 per frame, denoise in time (the grain on eyelids), motion-compensated 16 -> 30 fps, 738x1280 | box GPU + 32 CPU cores | ~7 s per clip in parallel |
| `render` | cuts, zoom/shake/flash, **glitch words** (a 4-frame hit then a clean word), hook title, branch tags, closing question, graded look (no grain), sound | **on the box** (`scripts/remote_render.py`; the PC only sends the plan, the voices and the clips the box does not have, and gets the mp4; `--render-on pc` to run it locally) | ~1 min on the box |
| `caption` | the text and the hashtags to post with the video | — | — |

The box app is switched by the script between the two families (`krea2` ↔ `s2v`), **one family at a time** (one GPU). A switch costs ~1 minute.

## One folder per TREND

Everything that is specific to a trend lives in `scripts/trends/<trend>.py`: the **constant prompts** of the agents (their words never change, only their `{{slots}}`, filled from the brief), `PROMPT_VERSION`
(recorded in every plan, with a fingerprint of each prompt), the structure recipe (how many shots each act has for N seconds), the JSON schemas of what each agent answers, the render preset. Another trend =
another file next to it; the chain, the pipeline and the checks do not change (`story_auto.py run NAME "..." --trend you_must_choose`).

**The library of approved stories** (`scripts/story_library.py`): `story_auto.py approve NAME` stores a story you like; the writer and the director get the two most similar approved stories as STYLE
examples. A story nobody approved is stored but never used as an example. The first entry is the story Claude wrote by hand for the first 2m30 video.

**Proof that it works** (`scripts/story_eval.py`): the same contexts are written by `chain`, `single` (one call) and `hand` (a person) and measured the same way (structure problems left, hook keywords, context words
kept, variety, repeated word pairs, pictures, endings, seconds); `--judge` adds a 1-10 score by the model on seven criteria. Reports in `results/story_trend/story_tests/`.

## Everything has a default drawn by the seed and can be overridden

`story_engine.resolve_params` writes, for every parameter, whether it was **given**, a fixed **default** or drawn by the **seed** (`provenance` in `plan.json`): language (`en`, `fr`), duration,
branches (1 or 2), tone, number of talking lines, **which branch ends badly and which well** (`endings`), voices, style. The style is stored in `results/story_trend/style_library` with its measured
quality (a style is reusable only if its pictures keep the same colour signature on at least two kinds of places). The spoken text, title, tags and closing question follow the language; picture
and motion prompts stay English (the image and video models are English). **All characters are fictional**: a real public figure in the context is replaced by an invented archetype.

A plan written elsewhere can be given with `--plan plan.json` (the story step is skipped); the format is the one of `scripts/story_writer.py` (`shots` with `kind`, `branch`, `speaker`, `text`, `location`,
`in_shot`, `still`, `motion`, `camera`, `fx`, optional `tag`, `time`, `ending`, `choice`). The first hand-written story, `results/story_trend/plans/the_last_city_150_en.json`, is also the model's example.

## What makes the faces good (learned the hard way — keep it)

* The S2V model draws at 480x832: a face that is 25 % of the frame has ~60 px: grainy eyes. **Use tight close-ups (face ≈ 40 % of the width, centred)** for everyone who speaks.
* LoRA `wan2.2_t2v_A14b_low_noise_lora_rank64_lightx2v_4step_1217` (lightx2v, public, no token): the calmest face (−24 % shimmer vs the v1 I2V LoRA), 4 steps = ~35 s per clip. `i2v_low_soft` (8 steps) is
  worse and 3x slower. Instareal (HF `Instara/instareal-wan-2.2`) helps the older LoRAs a little and nothing on this one.
* **Never add film grain** in the grade: it lands on the skin. The post pass removes the shimmer left between frames.
* Do **not** run ComfyUI in `normal`/`lowvram` mode with S2V + LoRA: CUDA OOM while patching the fp8 weights. `--novram` is the only mode that works.
* A *stacked* picture (two or three panels in one still) happens with Krea2 on long prompts: look at the contact sheet of the stills and redo those with `story_stills.py --only … --seed-shift 1000`.
* `scripts/s2v_face_test.py` compares settings on one face (numbers + a picture of the same frame); `scripts/eye_demo.py` makes a before/after video.

## The box

`scripts/box/setup_box.sh` (needs ≥ 60 GB RAM, ≥ 250 GB disk) copies the catalogue and starts the downloads; `fetch_extras.sh` the LoRAs and ESRGAN models; `setup_llm.sh` Ollama + the story model;
`start_app.sh` / `stop_app.sh` run the app. The model downloads are the real cost of a fresh box (~130 GB: 3 hours at 11 MB/s): pick a box with a fast link, or reuse the same box for several videos.
**Never close the box without the owner's word**; it bills while idle.

### What the image already contains (since `story-voice-20261009`)

Nothing of the audio / blink / LLM tooling is installed by hand on a new box any more. The image carries, where the pipeline looks for them:

| In the image | What | Used by |
|---|---|---|
| `/root/ttsenv` | torch 2.6 cu124, `qwen-tts` (Qwen3-TTS), `openai-whisper` + the Whisper `small` weights (`/root/.cache/whisper`) | `story_voices.py` (transcribe, speak, align) |
| `/root/eyelibs` | MediaPipe 0.10.14 + OpenCV + numpy < 2 (used with the python of `/root/ttsenv`) | `eyes_ear.py` (blink counter, the `qc` step) |
| `/usr/local/bin/ollama` | Ollama 0.40.2 (checksum-verified release); the model is pulled by `setup_llm.sh` | all the story agents |
| `/root/*.sh`, `/root/*.py` | the helpers of `scripts/box` (start/stop, downloads, voices, blink counter, upscaler) | the PC, over ssh |
| `/root/ttsenv.freeze.txt`, `/root/eyelibs.freeze.txt` | the exact package versions installed | reproducibility |

Still downloaded on a new box (they are weights, not tools): the model profiles and the three Qwen3-TTS repos (`download_all.sh`, Base + Tokenizer + VoiceDesign), the story model (`setup_llm.sh`), the extras (`fetch_extras.sh`).
**Not in the image on purpose:** the recorded voice samples (rights not verified, the `-public` twin is world-readable) and every token. The PC sends the samples itself before they are used
(`story_produce.ensure_voice_samples`: only the files the box lacks, ~1 MB).
The build proves the tools at build time (it imports each of them), so a missing one fails the build and not a rented hour.

## Test mode and automatic length (two options, neither is the default)

* **`--shot-scale 0.4`** (`story_auto.py run`, `story_agents.py`, `story_text_lab.py`; the parameter `shot_scale`, 1.0 = normal): TESTS ONLY. It multiplies the time a shot stays on screen, so that a short
  test holds MANY scenes: `budget()` plans more shots (a minute at 0.4 has as many shots as 2:30 at the normal pace, about 40), the lines get shorter (`max_words()`: 6 words instead of 14, offers 4 instead of 9), the silence
  after a line shrinks (`story_render.tail_of()`, never under 0.15 s; the twist keeps 0.96 s instead of 2.4 s, the choice keeps its 3.2 s countdown) and the clips are as short as their shot (about 1.5 s: fewer frames to
  generate). More shots also means more pictures to make (stills, identity pass), so the saving is on the length of the video and of every clip, not on the number of scenes. The plan records the scale it was made with.
* **`--auto-length`** (the parameter `length_mode: auto`): nobody fixes the length, the creation decides it, never under 2:00 and never over 10:00 (`LENGTH_RANGE`). The idea agent writes as many sentences as its story
  needs (5 to 14), then the LENGTH agent (`run_length`, prompt `LENGTH_PROMPT`) lists the key events of that context and gives a length (clamped, rounded to 10 s; without a model the length follows the size of the context).
  Everything after it follows the decided length: the analyst is asked for more places (up to 10 or 11), an outline of more than 60 beats is planned in 3 parts (up to the choice, branch A, rewind + branch B, each told what
  was planned before) because one answer of the model would be cut, and the lines are written in chunks of at most 28 beats. `story_text_lab.py --auto-length --ideas 0 --stories N --outline-candidates 1` writes the scripts
  (text only) with a summary of the lengths at the top of REPORT.md (shortest, longest, average, within the bounds, average at least 2:30).

## Checklist when something goes wrong

* `story_auto.py status NAME` says what is missing; rerun `run` — it resumes.
* The model never gives a valid story → the report says `template (the model never gave a valid story: [...])`: read the problems, they are the validator's.
* A shot says `NO CLIP` in the render log → its clip is missing (`animate` did not finish): rerun `--steps animate,post,render`.
* Black pictures from Qwen-Image-Edit → SageAttention is on (`COMFY_USE_SAGE_ATTENTION=0` for that family).
* `AudioEncoderLoader` lists nothing → the folder is not declared in `config/extra_model_paths.yaml` (a test now checks every catalogue folder).
* Voices of a recorded sample need its **exact transcript** and its **rights**: the three given voices look like a commercial library; private tests only until that is checked.
