# Test range: what to vary to improve every tool of the automatic pipeline

How it works: each line is an **experiment** = one tool, the settings to vary (the RANGE), what is measured by code, what you look at (you choose), and the cost. `scripts/motion_lab.py` runs the
experiments and puts the variants side by side in one video with their settings written under each panel. **Your choice is written to the methods registry** (`results/story_trend/methods/registry.json`,
`scripts/method_registry.py`): for every NEED of a shot (a person who speaks, who walks, hands in first person, a place, a two-person shot...) it keeps which method WORKS, which FAILS, and the proof, and
the pipeline picks the method by itself from the nature of the shot.

Status today: **[done]** measured, **[running]** in the lab now, **[to do]** not yet.

## What the user CHOSE after the first lab (2026-10-08) and what the pipeline now does with it
| need of a shot | method (registry) | the user's words |
|---|---|---|
| a person walks / runs | **I2V** `i2v_action` (or `i2v_follow` when the camera follows); `i2v_plain` is "average, lacks details" | Elara runs: C or D; Kael walks: B, C, D good |
| first-person hands | **I2V** `i2v_pov` (4 steps) or `i2v_pov6` (6 steps) | valve: B or C |
| the choice moment: both characters hold out a hand | **I2V** `i2v_offer_hands` (plan A, MANDATORY in every video); `i2v_offer_step` (plan B) kept as the alternative (`--offer-method i2v_offer_step`) | faces clean in A and B; the S2V plan C is less immersive: dropped for this scene |
| a person waits / blinks | **I2V** `i2v_idle` | Elara blinks on C and D only, Kael on all but C: confirmed my landmark counter |
| a character SPEAKS or shouts | **S2V** `s2v_voice` | "S2V only when the character speaks or shouts"; the reference video's author makes them speak very little, the narrator quotes their sentences: a character speaks only to give an ORDER (at most 4 talk shots) |
| a place without people | S2V + silence (proven) | - |
The pipeline reads these from `scripts/method_registry.py` (`route(shot)`); a new proof or choice is written with `python scripts/method_registry.py verdict NEED METHOD STATUS "note"`.

## 1. People who MOVE (the "they never walk" problem)
| tool | range | measured by code | you look at | cost |
|---|---|---|---|---|
| Wan I2V two experts (lightx2v LoRAs) vs S2V + silence [done, chosen] | engine S2V / I2V; I2V steps 4 / 6; LoRA strength 1.0 / 0.8; prompt style plain / action (steps, arms, weight) / camera follows; seconds 3 / 5; size 480x832 / 576x1024; 3 seeds | `motion_amount` (a frozen pose < 0.3), face stability along the clip | do the feet and arms move, does the camera follow, does the face stay the same | ~70 s per clip |
| I2V without LoRA (20 steps) [to do] | steps 12 / 20, cfg 3.5 / 5 | same | is the motion richer enough to be worth 5x the time | ~6 min per clip |
| FLF2V (first + last frame) [to do] | last frame = the same person 4 m further | same | controlled walking path | ~90 s |

## 2. The SAME face in every shot (character stability)
| tool | range | measured | you look at | cost |
|---|---|---|---|---|
| Qwen-Image-Edit-2511 identity pass [done, now a pipeline step; the wardrobe is in the prompt] | prompt short / strict; steps 4 / 8; Lightning LoRA on / off; megapixels 1.0 / 1.5; references: close-up only / close-up + sheet / two close-ups (two-shot); scene = place only / place with a generic person; 3 seeds | face detection found, face size, (to add: face-embedding distance to the close-up) | same face, hair, clothes as the reference | ~15 s per picture |
| Face embedding check [to do] | ArcFace-type embedding, threshold 0.35 / 0.45 | distance close-up vs shot, automatic retry with another seed above the threshold | - | CPU |
| Close-up as the start image of every talking / listening shot [done] | - | - | - | - |

## 3. Blinking and a living face
| tool | range | measured | you look at | cost |
|---|---|---|---|---|
| S2V prompt [done] | base / "blinks naturally every 2-3 s" / + negative "no blinking, frozen eyes"; 2 seeds | blinks counted ON THE BOX with face landmarks (eye aspect ratio, `scripts/box/eyes_ear.py`); a first pixel-contrast measure said 0 everywhere while the eyelids closed: dropped | natural rhythm, not a flutter | ~40 s |
| I2V without audio for listening shots [done] | same prompts | same | result: the eyes SHUT 0.9-1.5 s (not a blink) | ~70 s |
| Blink check after every talking clip [to do] | count blinks of each clip; none in a clip of 2.5 s or more -> make it again with another seed (max 2) | blinks (landmarks) | blinks are irregular: a short clip can have none | +40 s per retry |
| S2V without LoRA (20 steps) [to do] | steps 20, cfg 4.5-6 | same | does the base model blink more | ~3 min |
| Post blink insertion (eyelid warp) [to do, only if nothing above works] | blink every 3-5 s, 6 frames | - | not creepy | CPU |

## 4. The offer shot (both characters hold out a hand) and first-person ACTION
| tool | range | measured | you look at | cost |
|---|---|---|---|---|
| Two-shot with Qwen (two references) + I2V [done, chosen: plan A] | prompts: hands slowly toward you / they step forward; S2V with the first line | motion amount, both faces found | both hands come toward the camera, both faces stable | ~2 min |
| First-person hands (POV action) [done, chosen: B or C] | S2V vs I2V 4 / 6 steps; actions: turn a valve, push someone through a hatch, climb, pull a lever | motion amount | the hands really do the action, nothing trembles in place | ~70 s |
| Action beats written by the story agent [to do] | share of POV shots per branch 30 / 50 / 60 %; verbs list | share measured in the plan | variety of actions | text |

## 5. Camera, effects, rhythm
| tool | range | measured | you look at | cost |
|---|---|---|---|---|
| Shake [done in code, to approve] | style legacy / impact; amplitude 0 / 0.3 / 0.6 / 1.0; share of shaken shots 0 / 1 in 6 / 1 in 3 | regularity (autocorrelation peak), share | calm, never a regular wobble | CPU |
| Zoom, flash, glitch words [to do] | zoom 0.03-0.08; flash only twists / + shocks; glitch 2 / 4 / 6 frames; subtitle size 70 / 84 / 100 px | - | readable, not tiring | CPU |
| Shot length [to do] | tails 0.25 / 0.4 / 0.6 s; target 150 s with a hard cap of +10 % | total length | rhythm | CPU |

## 6. Pictures, faces after generation
| tool | range | measured | you look at | cost |
|---|---|---|---|---|
| WORLD of the pictures [done in code, test running] | style from the PLACES of the story (their tags), the hour and the atmosphere of the brief; the place written in the prompt; a shot's own `time` no longer turns the light to the sun (auto_ab_1: a sunlit stone arcade in a night station) | before / after sheet with the SAME seeds (`scripts/world_lab.py`) | same world in every picture, no sun at night | 14 s per picture |
| Picture judge (vision model on the box) [done in code, to run] | `scripts/still_judge.py`: sun in a night world, place that does not belong, action missing, collage; retry with another seed (max 2) in the `stills` step | verdict per picture in `run/stills_judge_N.json` | do the verdicts agree with your eyes | ~8 s per picture |
| Krea2 stills [done] | prompt: wide interior / tall low-angle; forbidden words cleaned; 3 seeds | stacked-panel detector [to add], text glyphs [to add] | no collage, no signs with gibberish | 14 s |
| Post pass [done] | ESRGAN x2 / x4plus; denoise 0.6 / 1.0 / 1.5; sharpen 0.35 / 0.5 / 0.8; interpolation mci / blend / none | face detail and shimmer | sharp, not waxy | ~7 s per clip |
| S2V LoRA profiles [done] | v1 / 1022 / t2v 1217 / + Instareal / 8 steps | face shimmer | winner: t2v 1217, 4 steps | ~40 s |

## 7. Voices and sound
| tool | range | measured | you look at | cost |
|---|---|---|---|---|
| Qwen3-TTS [to do] | clone vs designed; temperature 0.6 / 0.9; reference length; max tokens | transcription error rate (Whisper), duration against the text, pitch gender check | natural, distinct characters | ~1 s per line |
| Sound background [to do] | ambience level -34 / -31 / -28 dB; thunder 0 / 2 / 5; uplift on good ending | loudness | not tiring, voices clear | CPU |

## 8. The STORY (agents)
| tool | range | measured | you look at | cost |
|---|---|---|---|---|
| Structure [to do] | ONE offer scene where both hold out a hand, branches fully independent (no mention of the other), suspense at the end of each branch | checker: offer shots, other character's name absent in a branch, suspense verdict | the logic, the open ending | ~6 min of model |
| Model settings [to do] | temperature 0.6 / 0.8 / 1.0; outline candidates 1 / 2 / 3; best of 1 / 2 whole stories (judge); examples 0 / 2 approved stories; model qwen3.6:27b vs a smaller one | validator problems, judge score, repetitions, length error | the story | 6-18 min |
| Judge criteria [to do] | add: "is there suspense", "do the branches stay independent", "real action" | score | agreement with your taste | - |

## Order proposed (cheapest and most useful first)
1. [running] identity + motion + blink + offer + POV (the five complaints) -> you choose -> registry.
2. Story structure (offer scene, independent branches, suspense) as text, two seeds -> you choose.
3. Shake and shot length as short render comparisons.
4. The checks by code (face embedding, stacked panels, text glyphs) so that the pipeline retries by itself.
5. Voices and sound.
