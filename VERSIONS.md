# Versions épinglées

Date de validation statique : 2026-08-11. Patch release préparée : `0.5.1` (non construite et non poussée).

Cette combinaison a été vérifiée par audit de configuration, compilation Python et tests unitaires légers. Elle n’a pas été validée par un build Docker complet, un test GPU ou une génération Wan réelle dans l’environnement actuel.

Release de référence déjà construite et GPU-testée après hotfix manuel : `ghcr.io/leselyo/wan-sequence-cloud-v2:0.5.0-all-flux`, manifest list `sha256:6edfbc253deb5f13f4c133118e6ae3c57537a11c7f73411d47f6baae6c8ea340`, manifest AMD64 `sha256:8e840a3d9a1937a5963d1bec324bff00d6909adc0f8546a29e7473e22937957b`, config `sha256:94883fbc979e2d09b3b6dfa2ce21ab83da675cb7f5438b24fb0376b8d1281f34`. Cette image n’est pas modifiée par le présent patch.

| Composant | Version/révision |
|---|---|
| Plateforme cible | Linux AMD64 |
| Image CUDA | `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` |
| Manifeste AMD64 CUDA | `sha256:0bb88834d973ca1b450fcc2a05333c6fe45510bee289912a5391274c351c4a4d` |
| Ubuntu | 22.04 |
| Python | 3.10 fourni par Ubuntu 22.04 ; patch exact à confirmer après build |
| PyTorch | 2.5.1, wheels CUDA 12.4 |
| TorchVision | 0.20.1 |
| TorchAudio | 2.5.1 |
| ComfyUI | `2881e6161081439b1c3fb3b6c1f51b3d272da710` |
| Workflow templates | `1b3bdd46c945d54d893a3b43692d5963608fb7d4` |
| Wan 2.2 ComfyUI Repackaged | `fb1388adc906ab39ffc26ee40e96b22886b56bc4` |
| Wan 2.1 ComfyUI Repackaged | `06e001fc51048fb03433a6fb25334de7836704a5` |
| Comfy-Org FLUX.1-schnell | `f757664cb3ff8b19ff99e064a2387a5f547ad9e8` |
| FLUX text encoders | `2f74b39c0606dae3b2196d79c18c2a40b71f3250` |
| FLUX AE (Comfy-Org repackaged) | `817f3ba14d96a306cf0cbed49ef9eea037991381` |
| FastAPI | 0.115.12 |
| Uvicorn | 0.34.2 |
| HTTPX | 0.28.1 |
| Pydantic | 2.11.4 |
| python-multipart | 0.0.22, requis par `POST /v1/images/upload` |

| Format du manifest bundle | 1 |
| Schéma du catalogue LoRA | 2 |
| Taille de chunk OCI par défaut | 4 GiB (`4294967296` octets) |
| Schéma workflow converti | 4 |

Assets FLUX.1-schnell du profil `image` :

| Rôle | Fichier | Octets | SHA-256 |
|---|---|---:|---|
| diffusion | `diffusion_models/flux1-schnell.safetensors` | 23 782 506 688 | `9403429e0052277ac2a87ad800adece5481eecefd9ed334e1f348723621d2a0a` |
| CLIP-L | `text_encoders/clip_l.safetensors` | 246 144 152 | `660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd` |
| T5XXL FP8 | `text_encoders/t5xxl_fp8_e4m3fn.safetensors` | 4 893 934 904 | `7d330da4816157540d6bb7838bf63a0f02f573fc48ca4d8de34bb0cbfd514f09` |
| AE | `vae/ae.safetensors` | 335 304 388 | `afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38` |

Les modèles de base sont transportés sans conversion : le SHA-256 du fichier reconstruit doit être identique à celui du catalogue officiel. Les LoRA Hugging Face/Civitai restent dynamiques et ne font pas partie du bundle.

Profils bundle reconnus : `t2v`, `i2v`, `flf2v`, `all-video`, `image`, `flux-schnell`, `all` et les profils turbo historiques. `all-video` contient les six assets Wan dédupliqués. `image` contient les quatre fichiers FLUX.1-schnell natifs et `all` est leur union. `generate_start_image.engine=flux_schnell` est supporté par `image_flux_schnell.api.json`.

Custom nodes : aucun custom node n’est installé. Les workflows utilisent les nœuds natifs de la révision ComfyUI épinglée.

Pillow est réutilisé depuis les dépendances ComfyUI déjà installées dans le même environnement Python ; il n’est pas déclaré une seconde fois. Les modèles, commits ComfyUI, CUDA et PyTorch restent inchangés par le correctif 0.5.1.

Les tests de conversion conservent les trois templates de ce commit sous
`tests/fixtures/workflow_templates_1b3bdd46/`. T2V et I2V contiennent des
`definitions.subgraphs`; FLF2V n'en contient pas à cette révision.

Schéma des workflows API convertis : `4`. Un cache dépourvu de
`_converter_schema_version: 4`, même produit depuis le même workflow source,
est automatiquement régénéré au démarrage.
