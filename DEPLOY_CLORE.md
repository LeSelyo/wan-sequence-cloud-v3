# Construction et déploiement sur Clore.ai

## Machine de build

Le build est destiné à une machine Linux AMD64 disposant de Docker, Docker Buildx et d’une connexion stable. `MIN_BUILD_DISK_GB=100` n’est qu’un seuil de refus minimal, pas une promesse de capacité suffisante : le profil `all` transporte 93 424 945 275 octets, conserve ses chunks OCI et reconstruit les mêmes octets sous `/workspace`. Il faut dimensionner le disque à partir de l’espace réellement libre, des modèles manquants, de la réserve, des caches et des sorties. Aucun GPU n’est nécessaire pendant le build.

La machine de build et l’image applicative sont deux environnements distincts : Docker/Buildx restent sur la machine de build. L’image applicative ne contient ni Docker, ni LoRA, ni secret. La variante de production bundlée contient les chunks immuables des modèles, jamais les fichiers monolithiques reconstruits.

## Build et publication

Avant tout build coûteux, exécuter :

```bash
bash scripts/preflight-release.sh
```

Connexion préalable au registre avec `docker login`, puis :

```bash
IMAGE_REF=registre/utilisateur/wan-sequence:1.0.0 \
  ./scripts/build-and-push.sh
```

Fallback si le checkout local n'a pas matérialisé le bit exécutable :

```bash
IMAGE_REF=registre/utilisateur/wan-sequence:1.0.0 \
  bash scripts/build-and-push.sh
```

Cache de registre facultatif :

```bash
IMAGE_REF=registre/utilisateur/wan-sequence:1.0.0 \
BUILD_CACHE_REF=registre/utilisateur/wan-sequence:buildcache \
  ./scripts/build-and-push.sh
```

Le script construit et publie une seule image `linux/amd64`. Il refuse le build si un modèle ou média lourd est présent dans le contexte.

## Lancement sur une machine GPU

Pour les premiers essais, utiliser une location On-Demand afin d’éviter une interruption pendant l’installation ou la génération.

```bash
docker run --rm --gpus all \
  -p 8000:8000 \
  -v /workspace:/workspace \
  -e DATA_ROOT=/workspace \
  -e PORT=8000 \
  -e API_TOKEN="$API_TOKEN" \
  -e REQUIRE_API_TOKEN=1 \
  registre/utilisateur/wan-sequence:1.0.0
```

Exposer uniquement le port HTTP 8000. SSH n’est nécessaire que pour l’administration. ComfyUI reste sur `127.0.0.1:8188` dans le conteneur.

Le volume `/workspace` peut être vide ou déjà rempli. Au démarrage, l'entrypoint
crée les seuls sous-répertoires requis avec `appuser:appuser` et le mode `775`,
sans `chown -R` des modèles existants. Les chunks du profil bundle restent sous
`/opt/wan-model-parts`; l'entrypoint reconstruit automatiquement les modèles
manquants sous `/workspace/models` sans réseau. Les LoRA demandées par un job
peuvent, elles, être téléchargées depuis leur URL approuvée.

### Configuration Clore attendue

Le cold-start de la release 0.5.0 n’a pas abouti avant hotfix. L’image contenait au moins deux défauts déterministes : parents du bundle non traversables et entrypoint dépendant du CWD. Cela ne démontre pas que Clore avait ignoré la startup command.

- Container image : `ghcr.io/leselyo/wan-sequence-cloud-v2:<release>` ; Registry authentication désactivée si le package est public.
- SSH Autoinstall entrypoint : activé ; SSH authorization : clé publique.
- Ports : `22/TCP` et `8000/HTTP`. Ne pas exposer `8188`.
- Run a command inside the container on startup :

```bash
/app/scripts/entrypoint.sh
```

- Environment : `DATA_ROOT=/workspace`, `PORT=8000`, `COMFYUI_HOST=127.0.0.1`, `COMFYUI_PORT=8188`, `GPU_CONCURRENCY=1`, `READINESS_PROFILE=all`, `REQUIRE_API_TOKEN=1`, `API_TOKEN=<secret>`.
- `CIVITAI_API_TOKEN` est **requis** pour télécharger n'importe quelle LoRA CivitAI du catalogue (l'API renvoie 401 sans token) ; `HF_TOKEN` n'est nécessaire que pour des ressources HuggingFace privées/gated (les modèles Wan/Flux du catalogue n'en demandent pas). `API_TOKEN` n’est pas le PAT GHCR et aucun de ces secrets ne doit entrer dans Git ou l’image.
- Films longs (ex. 12 plans FLF2V de ~5 s) : les défauts refusent le job (`MAX_SHOTS_PER_JOB=8`, `MAX_TOTAL_FRAMES=968`, `MAX_JOB_COST_UNITS=8`). Coût d'un plan = largeur×hauteur×images×steps / (832×480×121×20) ; relever ces trois variables avec de la marge (ex. `MAX_SHOTS_PER_JOB=20`, `MAX_TOTAL_FRAMES=2400`, `MAX_JOB_COST_UNITS=20`). `scripts/film_client.py` rejoue tout le scénario (keyframes Flux + LoRA, FLF2V, concat ffmpeg) via l'API bearer ; `scripts/smoke_concat.py` valide ffmpeg dans l'image.
- **Wan 2.2 Animate (profil `wan22-animate`, modes `animate_mix`/`animate_move`)** : ajoute ~21,8 Go de modèles (diffusion fp8 18,4 Go + LoRA turbo 480p 0,74 Go + LoRA de relighting 1,44 Go + CLIP Vision 1,26 Go), en plus de `umt5_xxl`/`wan21_vae` déjà partagés. Nécessite 3 paquets de noeuds ComfyUI tiers, installés et figés par commit dans l'image (`comfyui_controlnet_aux`, `ComfyUI-KJNodes`, `ComfyUI-segment-anything-2`) — aucun n'existait dans l'image avant cet ajout. Les bindings (dont le point SAM2 et `background_inputs` pour le mode Move) ont été validés en conditions réelles le 2026-10-02 sur RTX 4090 ; `prepare_workflows.py` garde néanmoins le workflow en mode tolérant (`EXPERIMENTAL`). Après une mise à jour de l'image, relancer `prepare_workflows.py --force` si un `wan22_animate.api.json` plus ancien est en cache (`CONVERTER_SCHEMA_VERSION` est maintenant 5). **Hôte à moins de 48 Go de RAM** (ex. 32 Go) : `--novram` range tous les poids en RAM et fait tuer ComfyUI par le noyau (parfois l'hôte entier se fige) ; `--lowvram` seul échoue en OOM GPU au patch LoRA des poids fp8. L'entrypoint choisit alors tout seul `--lowvram --reserve-vram 9` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (pic RSS ~26 Go), sauf si `COMFY_VRAM_MODE` est défini ; `COMFY_AUTO_LOWRAM=0` désactive ce choix, `COMFY_LOW_RAM_THRESHOLD_GIB` règle le seuil et `COMFY_EXTRA_ARGS` ajoute des options libres. Redémarrer ComfyUI entre familles de modèles (Krea2, FLF2V, Animate) sur ces machines. Préférer un hôte ≥ 64 Go quand c'est possible.
- **Krea 2 (moteur d'image `krea2`, profil `krea2`)** : alternative à Flux Schnell pour les images clés (`generate_start_image.engine: "krea2"` ou `POST /v1/images` avec `engine: "krea2"`), avec ses propres LoRA de style (répertoire `models/loras`, même mécanisme que les LoRA existantes). Ajoute ~18,6 Go de modèles (diffusion turbo fp8 13,1 Go + encodeur de texte Qwen3-VL-4B 5,2 Go + VAE 0,25 Go), depuis le dépôt officiel `Comfy-Org/Krea-2`. **Aucun noeud tiers** : `CLIPLoader(type="krea2")` est un type déjà natif à la révision ComfyUI figée (vérifié dans le code source, pas supposé). Attention : le workflow officiel `api_krea2_t2i.json` de Comfy-Org appelle en réalité une **API hébergée** (noeud `Krea2ImageNode`), pas un modèle auto-hébergé — c'est `image_krea2_turbo_t2i.json` (sans le préfixe `api_`) qui contient le vrai graphe local, celui utilisé ici comme référence. Le pas par défaut réel de Krea 2 Turbo est **8** (pas 4 comme Flux Schnell), confirmé dans ce fichier officiel. La capacité "Identity Edit"/référence de personnage n'est **pas encore implémentée** (seulement le texte-vers-image de base) — son mécanisme réel reste à vérifier avec la même rigueur.
- **Wan 2.2 VACE (profil `wan22-vace`, mode `vace`)** : plusieurs images clés à des positions temporelles choisies, pas juste début/fin. Ajoute ~34,7 Go de modèles (deux UNet fp8 17,35 Go, variante "Fun" du même dépôt officiel `Comfy-Org/Wan_2.2_ComfyUI_Repackaged` que t2v/i2v/flf2v), en plus de `umt5_xxl`/`wan21_vae` déjà partagés. **Aucun noeud tiers** : `WanVaceToVideo` est natif de ComfyUI (confirmé dans le code source à la révision figée). L'app construit elle-même, en Python (`build_vace_control_assets`, testé localement avec un vrai ffmpeg), les deux vidéos que `WanVaceToVideo` attend réellement — un `control_video` avec chaque image clé posée à son image exacte et un remplissage gris au milieu, et un `control_masks` assorti (blanc = à générer, noir = garder l'image clé telle quelle) — parce que ce noeud n'a pas d'entrée simple du genre "liste d'images + positions". Le fichier `workflows/wan22_vace.api.json` est écrit à la main (comme `image_flux_schnell.api.json`), à partir du code source réel du noeud et du gabarit haute/basse fréquence déjà validé pour i2v ; **son câblage n'a pas encore été confronté à un vrai ComfyUI**.
- **FFmpeg récent et plan `jumpcut`** : l'image installe un FFmpeg statique 8.1 dans `/opt/ffmpeg` (`FFMPEG_BIN`/`FFPROBE_BIN`, en tête du `PATH` y compris via `entrypoint.sh`) ; le paquet Ubuntu 4.4.2 n'est qu'un repli. Après le premier démarrage, vérifier : `curl -s -H "Authorization: Bearer $API_TOKEN" http://localhost:8000/v1/montage` doit montrer `"ready": true`, une version `8.1…` et `"missing": []`. Un plan `jumpcut` ne consomme pas de GPU (voir README, « Montage FFmpeg »). Cette image n'a pas encore été construite ni lancée avec ce changement : le build échoue exprès si un filtre ou `libx264` manque, et `--build-arg FFMPEG_SHA256=…` épingle le binaire.
- Quitter SSH ne stoppe pas la facturation : arrêter/terminate explicitement l’ordre Clore (site : *My orders* → cocher la commande → *Cancel selected* → *Yes, cancel all* ; irréversible, copier d'abord les résultats ; API : `POST /v1/cancel_order` avec l'en-tête `auth: <clé>`, pas `Bearer`). Un serveur inactif coûte ~0,44 $/h pour une 4090.

Le script accepte un lancement initial en `root`, crée uniquement les chemins
runtime requis, leur donne `appuser` comme propriétaire puis se relance avec
`gosu`. Il exporte `/opt/venv/bin` et appelle directement
`/opt/venv/bin/python` et `/opt/venv/bin/uvicorn`; il ne dépend donc pas du
`PATH` injecté par l'hébergeur. Les dossiers ComfyUI explicitement inscriptibles
sont `/opt/ComfyUI/user` et `/opt/ComfyUI/temp`; le reste de l'installation
ComfyUI conserve ses droits de lecture normaux.

Le script fixe explicitement son `APP_ROOT`, change vers `/app` et lance Uvicorn avec `--app-dir /app`; son comportement ne dépend donc plus du CWD choisi par l’orchestrateur. Il journalise uniquement l’état `API_TOKEN=set|unset`, jamais sa valeur. Le préflight du bundle est exécuté sous `appuser` avant matérialisation.

Les tokens `HF_TOKEN` et `CIVITAI_API_TOKEN` ne sont transmis qu'au runtime pour les sources LoRA qui exigent une authentification. `HF_TOKEN` reste également accepté par le downloader manuel de modèles, utile pour les profils non embarqués :

```bash
docker exec -e HF_TOKEN="$HF_TOKEN" CONTAINER \
  /opt/venv/bin/python /app/scripts/download_base_models.py --profile all --estimate

docker exec -e HF_TOKEN="$HF_TOKEN" CONTAINER \
  /opt/venv/bin/python /app/scripts/download_base_models.py --profile all

docker exec -e CIVITAI_API_TOKEN="$CIVITAI_API_TOKEN" CONTAINER \
  /opt/venv/bin/python /app/scripts/download_loras.py \
  pixel_gamegirl_wan22_t2v_high pixel_gamegirl_wan22_t2v_low
```

Vérification sans téléchargement :

```bash
docker exec CONTAINER /opt/venv/bin/python /app/scripts/download_base_models.py --check
docker exec CONTAINER /opt/venv/bin/python /app/scripts/download_loras.py --check
```

Pour vérifier uniquement les dépendances du profil réellement validé T2V :

```bash
docker exec CONTAINER /opt/venv/bin/python \
  /app/scripts/download_base_models.py --profile t2v --check
```

`--estimate` n'utilise pas le réseau et ne télécharge rien. Pour limiter les coûts, préférer `--profile t2v`, `i2v` ou `flf2v` à `all`. Les fichiers UMT5/VAE déjà valides sont réutilisés entre profils.

La rétention est manuelle et sèche par défaut :

```bash
docker exec CONTAINER /opt/venv/bin/python /app/scripts/cleanup_jobs.py \
  --older-than-hours 168 --status completed --dry-run
```

Ajouter `--force` uniquement après vérification de cette liste. Le script ne cible jamais modèles, LoRA, caches de modèles, workflows ou manifestes.

## Tests HTTP

Une fois le conteneur démarré, le smoke test non-inférentiel vérifie les
exécutables absolus avec un `PATH` minimal, les droits effectifs de `appuser`,
les trois fichiers API et leurs bindings, puis les deux routes de santé :

```bash
docker exec CONTAINER /app/scripts/smoke_runtime.sh
```

Il ne télécharge ni modèle ni LoRA et ne génère aucune vidéo. La route
`/health/ready` peut légitimement répondre `503` si le profil demandé exige des
modèles absents ; le script vérifie alors la structure détaillée de la réponse,
sans masquer cette absence.

Le smoke vérifie également que chaque `SaveVideo` utilise `mp4/auto` et qu'aucun
nœud runtime ne référence les LoRA Lightning optionnelles des templates
officiels. Ces LoRA ne sont pas nécessaires au chemin T2V/I2V standard exposé
par cette API ; les LoRA utilisateur sont injectées par le backend sur le chemin
modèle commun.

```bash
curl http://127.0.0.1:8000/health/live
curl http://127.0.0.1:8000/health/ready
curl -H "Authorization: Bearer $API_TOKEN" \
  http://127.0.0.1:8000/v1/catalog
```

Création d’un job :

```bash
curl -X POST http://127.0.0.1:8000/v1/jobs \
  -H "Authorization: Bearer $API_TOKEN" \
  -H 'Content-Type: application/json' \
  --data-binary @/app/examples/sequence.json
```

Les données sont sous `/workspace`. Si la persistance de la location n’est pas garantie, sauvegarder modèles, manifestes, base SQLite et résultats avant sa fin.

## Mode backend sans GPU

```bash
docker run --rm -p 8000:8000 \
  -v "$PWD/test-data:/workspace" \
  -e APP_TEST_MODE=1 \
  -e API_TOKEN=test-token \
  IMAGE_REF
```

Ce mode ne lance pas ComfyUI et crée uniquement une petite sortie factice.

## Déploiement d'une image avec modèles vidéo embarqués

La préparation lourde se fait sur la machine de build, jamais dans un stage Docker :

`Comfy-Org épinglé → validation → split 4 GiB → chunks → layers OCI distinctes → GHCR`.

```bash
BUNDLED_MODEL_PROFILE=all \
IMAGE_REF=ghcr.io/OWNER/wan-sequence-cloud:bundle-all-flux-test \
bash scripts/build-and-push.sh
```

Après le pull, l'entrypoint crée les répertoires appuser puis reconstruit automatiquement les fichiers monolithiques sous `/workspace/models`. ComfyUI ne voit jamais les chunks et continue de charger les mêmes `.safetensors`. Prévoir simultanément l'espace occupé par les layers de l'image et celui des fichiers reconstruits dans le volume `/workspace`, plus 1 GiB de réserve par défaut.

`t2v`, `i2v` et `flf2v` embarquent seulement leur famille ; `all-video` embarque l’union dédupliquée T2V/I2V/FLF2V. `image`/`flux-schnell` embarque les quatre assets FLUX.1-schnell. `all` contient les six assets Wan plus FLUX diffusion, CLIP-L, T5XXL FP8 et AE. `READINESS_PROFILE=all-video` n’exige pas FLUX ; `READINESS_PROFILE=all` exige également `image_flux_schnell.api.json` et les quatre fichiers image vérifiés.

`all` représente 93 424 945 275 octets (87,01 Gio) de modèles reconstruits. Les chunks de transport et les fichiers monolithiques sous `/workspace` coexistent : budgéter au moins environ 174,02 Gio pour ces deux copies, puis ajouter l’image applicative, la réserve, les caches et les sorties.

Les images FLUX sont des jobs asynchrones sous `/workspace/outputs/images/<image_id>/`. Le client peut récupérer le PNG ou transmettre `start_image.image_id` à un job Wan I2V. `generate_start_image.engine=flux_schnell` enchaîne automatiquement FLUX puis I2V. Une image externe continue de contourner entièrement FLUX.

Les LoRA restent sous `/workspace/models/loras` et sont téléchargées à la demande uniquement depuis l'URL approuvée du catalogue. LightX2V reste également dynamique. Fournir `HF_TOKEN` ou `CIVITAI_API_TOKEN` au runtime seulement si la source le demande. Un volume monté sur `/workspace` ne masque jamais les chunks de transport situés sous `/opt/wan-model-parts`.

Le cache registre reste strictement optionnel. Lorsque `BUILD_CACHE_REF` est absent ou vide, `build-and-push.sh` ne passe aucun `--cache-to type=registry`; le cache BuildKit local reste utilisable. L’échec historique connu s’est produit pendant l’export du cache GHCR, sans preuve suffisante pour attribuer une cause plus précise.

La release privée `ghcr.io/leselyo/wan-sequence-cloud-v2:0.5.0-all-flux` et sa copie publique ne doivent pas être remplacées. Le prochain tag prévu est une nouvelle patch release.

Commande future, uniquement après validation du mini-test Docker sur un hôte possédant un daemon fonctionnel :

```bash
BUNDLED_MODEL_PROFILE="all" \
IMAGE_REF="ghcr.io/leselyo/wan-sequence-cloud-v2:0.5.1-all-flux" \
MIN_BUILD_DISK_GB="100" \
bash scripts/build-and-push.sh
```

Ne définir aucun `BUILD_CACHE_REF` par défaut. Cette commande n’a pas été exécutée pendant le correctif.
