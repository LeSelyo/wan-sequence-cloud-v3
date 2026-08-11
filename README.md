# WAN Sequence Cloud

API Docker GPU pour exécuter une séquence arbitraire de plans Wan 2.2, puis les assembler en MP4.

Modes du payload :

- `t2v` : texte vers vidéo, avec Wan 2.2 T2V ;
- `t+i2v` : texte + image initiale vers vidéo, avec Wan 2.2 I2V ;
- `i2v` : image initiale vers vidéo, le prompt est facultatif ;
- `t+i(keyframe)2v` : texte + première et dernière images vers vidéo ;
- `i(keyframe)2v` : première et dernière images vers vidéo, le prompt est facultatif.

Le service valide chaque LoRA avant d’envoyer le workflow à ComfyUI. Une LoRA SD 1.5, Illustrious ou Pony ne peut jamais être injectée dans Wan. Les éléments de votre liste qui ne sont pas des LoRA sont absents du catalogue exécutable.

## Architecture

Le conteneur lance :

1. ComfyUI natif comme moteur GPU local ;
2. une API FastAPI sur le port `8000` ;
3. une file GPU à concurrence 1 ;
4. FFmpeg pour concaténer les plans ou produire un fondu.

Les workflows officiels Wan 2.2 T2V, I2V et FLF2V sont récupérés à une révision épinglée pendant le build, puis convertis au format API au démarrage. Le convertisseur développe les sous-graphes T2V/I2V avant de produire le prompt API ; le template FLF2V épinglé reste un graphe principal. La conversion conserve une validation stricte des bindings et normalise `SaveVideo` en MP4. Les LoRA Lightning optionnelles incluses dans les templates T2V/I2V sont contournées sur le chemin standard exposé par l'API ; elles ne sont pas requises dans `/workspace/models/loras`. Les LoRA utilisateur restent injectées séparément après le switch modèle.

`DATA_ROOT` vaut `/workspace` par défaut et contient modèles, entrées, résultats, base SQLite, caches et manifestes. Au démarrage root, le conteneur crée idempotemment tous ses sous-répertoires et ajuste uniquement leurs propriétaires et modes, sans parcourir récursivement les modèles existants. L'image standard reste sans modèles ; une variante bundlée transporte des chunks immuables sous `/opt/wan-model-parts` et les matérialise automatiquement sous `/workspace/models`.

## Matériel conseillé

La configuration native 14B est lourde. Pour un déploiement simple, prévoir idéalement un GPU NVIDIA de 48 Go de VRAM et au moins 64 Go de RAM. Le T2V FP8 peut fonctionner plus bas avec offload, mais le temps et la mémoire hôte augmentent. Le workflow GGUF 8 Go cité dans la demande n’est pas intégré : c’est un workflow, pas une LoRA, et il dépend de plusieurs nœuds communautaires.

## Construction et lancement

Prérequis : Docker, Docker Compose, pilote NVIDIA et NVIDIA Container Toolkit.

Copier `.env.example` vers `.env` si des jetons de téléchargement sont nécessaires.

```bash
docker compose build
docker compose run --rm --entrypoint /opt/venv/bin/python wan-sequence \
  /app/scripts/download_base_models.py --profile all --estimate
docker compose run --rm --entrypoint /opt/venv/bin/python wan-sequence \
  /app/scripts/download_base_models.py --profile all
docker compose run --rm --entrypoint /opt/venv/bin/python wan-sequence /app/scripts/download_loras.py \
  pixel_gamegirl_wan22_t2v_high pixel_gamegirl_wan22_t2v_low \
  spatial_magic_v2_wan22_i2v_high
docker compose up
```

Pour CivitAI, définir si nécessaire `CIVITAI_API_TOKEN`. Pour Hugging Face, définir si nécessaire `HF_TOKEN`. Ne jamais incorporer ces secrets dans le Dockerfile.

Les révisions sont épinglées dans le Dockerfile et documentées dans [VERSIONS.md](VERSIONS.md). Le Dockerfile applicatif ne télécharge aucun modèle. Pour une variante bundlée, `prepare_bundled_models.py` télécharge et vérifie les poids sur l’hôte avant le build, puis le Dockerfile généré transporte uniquement leurs chunks. Les LoRA restent hors bundle.

Vérification locale sans téléchargement :

```bash
docker compose run --rm --entrypoint /opt/venv/bin/python wan-sequence \
  /app/scripts/download_base_models.py --check
docker compose run --rm --entrypoint /opt/venv/bin/python wan-sequence \
  /app/scripts/download_loras.py --check
```

Un fichier valide est réutilisé. `--force` est nécessaire pour imposer son remplacement.

Sur Clore, la vérification T2V équivalente est :

```bash
/opt/venv/bin/python /app/scripts/download_base_models.py --profile t2v --check
/opt/venv/bin/python /app/scripts/download_base_models.py --profile t2v-turbo --check
```

Le téléchargement des modèles exige une sélection explicite : `--profile t2v|t2v-turbo|i2v|i2v-turbo|flf2v|all-video|image|flux-schnell|all|all-turbo` ou `--ids ID...`. Les profils standard n'incluent aucune LightX2V ; les profils `*-turbo` ajoutent uniquement les paires high/low officielles correspondantes. `--list` affiche le catalogue, `--estimate` calcule les octets attendus, valides et restants sans réseau, et `--check` vérifie taille exacte puis SHA-256. Les fichiers partagés (UMT5 et VAE) ne sont comptés et téléchargés qu'une fois. Les tailles et SHA-256 viennent des pointeurs LFS/Xet officiels épinglés ; la readiness se limite volontairement à présence, taille exacte et manifeste pour éviter de relire plusieurs dizaines de Go.

## Entrées et appel API

Placer les images locales dans `data/inputs/`. Le chemin d’un payload est relatif à ce dossier. Les URL HTTP(S) publiques sont aussi acceptées ; les adresses privées et loopback sont bloquées.

```bash
curl -X POST http://localhost:8000/v1/jobs \
  -H "Authorization: Bearer $API_TOKEN" \
  -H 'Content-Type: application/json' \
  --data-binary @examples/sequence.json

curl -H "Authorization: Bearer $API_TOKEN" \
  http://localhost:8000/v1/jobs/demo_sequence_001
curl -H "Authorization: Bearer $API_TOKEN" -o sequence.mp4 \
  http://localhost:8000/v1/jobs/demo_sequence_001/output
```

Le catalogue réellement utilisable est visible via :

```bash
curl -H "Authorization: Bearer $API_TOKEN" http://localhost:8000/v1/catalog
```

## Structure importante du payload

```json
{
  "id": "ma_sequence",
  "transition": {"type": "cut"},
  "shots": [
    {
      "id": "plan_01",
      "mode": "t2v",
      "prompt": "Pixel style, retro video game, ...",
      "width": 832,
      "height": 480,
      "frames": 121,
      "fps": 24,
      "seed": 42,
      "steps": 20,
      "cfg": 5.0,
      "turbo_mode": false,
      "loras": [
        {"id": "pixel_gamegirl_wan22_t2v_high", "weight": 0.9},
        {"id": "pixel_gamegirl_wan22_t2v_low", "weight": 0.9}
      ]
    }
  ]
}
```

Pour Wan 2.2 MoE, une paire de fichiers high-noise et low-noise doit être déclarée comme deux entrées distinctes. `target: auto` utilise la cible enregistrée dans le catalogue. Un poids explicite reste ajustable par plan.

`turbo_mode` vaut `false` par défaut. En T2V/I2V, `true` matérialise les deux LightX2V officielles et impose les valeurs du fixture épinglé : `steps=4`, `cfg=1.0`, split `2`. Une valeur `steps`/`cfg` explicite incompatible est refusée. Les LoRA utilisateur restent appliquées après LightX2V. FLF2V ne prend pas en charge ce mode.

## Images FLUX.1-schnell et T2I → I2V

Le moteur image de production est `flux_schnell`. Le workflow natif `image_flux_schnell.api.json` utilise `UNETLoader`, `DualCLIPLoader`, `VAELoader`, `CLIPTextEncodeFlux`, `KSampler`, `VAEDecode` et `SaveImage`. FLUX Schnell accepte 1 à 4 étapes ; `cfg` reste fixé à `1.0`. L’API expose `guidance`, mais refuse un negative prompt non vide plutôt que de l’ignorer.

```bash
curl -X POST http://localhost:8000/v1/images \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  --data @examples/image-flux-schnell.json
curl -H "Authorization: Bearer $API_TOKEN" http://localhost:8000/v1/images/img_ID
curl -H "Authorization: Bearer $API_TOKEN" \
  -o image.png http://localhost:8000/v1/images/img_ID/output
```

`POST /v1/images` retourne `image_id`, `status`, `status_url` et `output_url`. Le PNG et `request.json` sont conservés sous `/workspace/outputs/images/<image_id>/`. Pour réutiliser l’image sans téléchargement/réupload, fournir `start_image: {"image_id":"img_..."}` à `/v1/jobs`. La référence est résolue uniquement dans ce stockage contrôlé.

Le standalone accepte désormais `loras: []`. Une entrée non vide est réellement validée, téléchargée par le mécanisme dynamique existant puis injectée dans le workflow, mais seulement si son catalogue déclare explicitement sa compatibilité `flux_schnell`. Une LoRA Wan/SD/Pony incompatible est refusée avant ComfyUI.

### Upload direct d’une image utilisateur

`POST /v1/images/upload` reçoit un fichier multipart authentifié. PNG et JPEG sont décodés par Pillow, contrôlés en taille et pixels, corrigés selon EXIF puis réencodés en PNG canonique. Le nom et le Content-Type fournis par le client ne déterminent jamais le chemin ni le format accepté. La réponse fournit un `image_id` immédiatement `completed`, réutilisable comme `start_image` ou `end_image`. Limites par défaut : `MAX_INPUT_DOWNLOAD_MB=50` et `MAX_INPUT_IMAGE_PIXELS=16777216`.

```powershell
$BASE = "https://<clore-http-url>"
$TOKEN = "<API_TOKEN>"

curl.exe "$BASE/health/live"
curl.exe "$BASE/health/ready"

# Doit répondre 401 lorsque REQUIRE_API_TOKEN=1
curl.exe -i "$BASE/v1/catalog"

curl.exe -i `
  -H "Authorization: Bearer $TOKEN" `
  "$BASE/v1/catalog"

$UPLOAD = curl.exe -sS -X POST `
  -H "Authorization: Bearer $TOKEN" `
  -F "file=@C:\Users\user\Pictures\image.png" `
  "$BASE/v1/images/upload" | ConvertFrom-Json

$IMAGE_ID = $UPLOAD.image_id
```

Le payload I2V/FLF2V utilise ensuite :

```json
{"start_image":{"image_id":"img_..."}}
```

Routes publiques de l’API : `GET /health/live`, `GET /health/ready`, puis les routes authentifiées `GET /v1/catalog`, `POST /v1/images`, `POST /v1/images/upload`, `GET /v1/images/{image_id}`, `GET /v1/images/{image_id}/output`, `POST /v1/jobs`, `GET /v1/jobs/{job_id}` et `GET /v1/jobs/{job_id}/output`. ComfyUI sur `8188` n’est pas une route cliente publique.

Pour le pipeline automatique, utiliser `generate_start_image.engine=flux_schnell`. Si `start_image` est présent, FLUX n’est jamais appelé. Si les deux champs sont présents, la requête est refusée. Si aucun n’est présent pour I2V, la validation échoue avant ComfyUI. Voir `image-flux-schnell.json`, `sequence-i2v-generated-image.json`, `sequence-i2v-external-image.json` et `sequence-generate-start-image.json` sous `examples/`.

## Persistance, santé et sécurité

- Monter un disque persistant sur `DATA_ROOT` (`/workspace` par défaut).
- Exposer uniquement le port `8000`. ComfyUI écoute sur `127.0.0.1:8188` dans le conteneur.
- Définir `API_TOKEN` pour protéger toutes les routes `/v1/*`. `REQUIRE_API_TOKEN=1` refuse un démarrage non protégé.
- Utiliser un seul worker Uvicorn par GPU. La file interne sérialise les générations.
- Les jobs sont conservés dans `${DATA_ROOT}/jobs/jobs.sqlite3`. Un job `queued` ou `running` au redémarrage devient `interrupted`.
- Les vidéos sont sous `${DATA_ROOT}/outputs/<job_id>/`; les image jobs sous `${DATA_ROOT}/outputs/images/<image_id>/`.

```bash
curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready
```

Les limites par défaut sont 8 plans, 2 jobs en attente/exécution, 968 images totales, 50 Gio de sorties, 10 Gio libres et 8 unités de coût relatif. Une unité correspond à `832×480×121×20`; ce n'est pas un prix monétaire.

`READINESS_PROFILE=backend` vérifie l'API, SQLite et les répertoires sans exiger ComfyUI ou les modèles. Les profils `t2v`, `i2v`, `flf2v`, `all-video`, `image`, `flux-schnell` et `all` sont acceptés. `t2v-turbo`, `i2v-turbo` et `all-turbo` exigent aussi les LightX2V concernées, sans chargement VRAM.

La rétention n'est jamais automatique. La première commande ne fait qu'énumérer les jobs concernés ; aucune suppression n'a lieu sans `--force` :

```bash
/opt/venv/bin/python /app/scripts/cleanup_jobs.py --older-than-hours 168 --status completed --dry-run
/opt/venv/bin/python /app/scripts/cleanup_jobs.py --older-than-hours 168 --status completed --force
```

Elle peut retirer uniquement la sortie, les temporaires et copies d'entrées appartenant au job, puis sa ligne SQLite. Elle ne touche jamais aux modèles, LoRA, caches de modèles, workflows ou manifestes.

## Mode sans GPU

`APP_TEST_MODE=1` démarre FastAPI et SQLite sans lancer ComfyUI. Les faux jobs passent par `queued`, `running`, puis `completed` avec une petite sortie factice. Ce mode n’est jamais actif par défaut.

## Preflight obligatoire avant une release bundlée

La commande unique suivante compile Python, valide les JSON et Bash, importe l’API, exécute toute la suite, génère un faux profil `all` de dix modèles via le vrai générateur, matérialise deux fois les faux chunks et tente une mini-image Docker lorsque le daemon est disponible. Elle ne télécharge aucun vrai modèle :

```bash
bash scripts/preflight-release.sh
```

Un échec critique retourne un code non nul. Si Docker n’est pas disponible, le rapport indique explicitement `Mini Docker image: NOT RUN — Docker unavailable`; les contrôles statiques, API, CWD et matérialisation restent exécutés.

## Tests rapides

```bash
python -m unittest discover -s tests -v
python -m compileall -q app scripts
docker compose config
```

## Modèles embarqués et LoRA à la demande

Le build de publication utilise `BUNDLED_MODEL_PROFILE=t2v` par défaut : catalogue officiel → téléchargement résumable sur l'hôte → validation taille/SHA-256 → découpage binaire de 4 GiB → une instruction `COPY --link` et une layer OCI par chunk → push GHCR. Le modèle monolithique temporaire reste sous `.bundled-models/work`, chemin exclu du contexte Docker, puis il est supprimé après validation des chunks. Il ne peut donc entrer ni dans l'image ni dans le cache registry `mode=max`.

Profils de base :

- `t2v` : high/low T2V, UMT5 et VAE ;
- `i2v` : high/low I2V, UMT5 et VAE ;
- `flf2v` : high/low I2V, UMT5 et VAE requis par FLF2V ;
- `all-video` : union dédupliquée des six fichiers vidéo ;
- `image`/`flux-schnell` : diffusion FLUX Schnell, CLIP-L, T5XXL FP8 et AE ;
- `all` : `all-video` plus les quatre assets FLUX.1-schnell.

Le profil `image` représente 29 257 890 132 octets. `all` représente 93 424 945 275 octets (environ 87,01 Gio) de fichiers finaux. Les chunks restent dans les layers et les fichiers reconstruits occupent de nouveau cet espace dans `/workspace` : prévoir au moins environ 174,02 Gio pour ces deux copies, puis l’image applicative, la réserve, les caches et sorties.

La readiness retourne `capabilities.t2v`, `capabilities.i2v`, `capabilities.flf2v` et `capabilities.image_generation`, avec le détail des modèles et workflows. `all-video` n’exige pas FLUX. `image` exige le workflow et les quatre assets FLUX vérifiés. `all` exige les trois familles Wan et `image_generation`.

Au démarrage, `/app/scripts/materialize_bundled_models.py` lit `/opt/wan-model-parts/manifest.json`, vérifie les chunks, contrôle l'espace libre, reconstruit chaque `.safetensors.part`, valide son SHA-256 puis effectue un `os.replace()` atomique sous `/workspace/models`. Un fichier final déjà valide est réutilisé sans écriture. Les chunks immuables et les modèles reconstruits consomment tous deux de l'espace disque.

Les LoRA ne sont pas embarquées. Un job fournit seulement un ID approuvé du catalogue. Si le fichier manque, sa source et son URL explicites `huggingface` ou `civitai` sont utilisées avec reprise, validation et verrou par fichier. `HF_TOKEN` et `CIVITAI_API_TOKEN` sont optionnels et restent uniquement dans les headers mémoire. LightX2V suit ce même flux lorsque `turbo_mode=true`; le mode standard ne déclenche aucun téléchargement LightX2V.

Premier build de bundle :

```bash
BUNDLED_MODEL_PROFILE=all \
IMAGE_REF=ghcr.io/OWNER/wan-sequence-cloud:bundle-all-flux-test \
bash scripts/build-and-push.sh
```

## Limites connues

- Le premier build du bundle télécharge et découpe plusieurs dizaines de Go ; cette opération n'est pas exécutée par les tests locaux.
- Sur RTX 4090, FLUX standalone, Wan T2V standard, Wan I2V avec `image_id` et I2V avec `generate_start_image` ont été validés sur la release 0.5.0 après hotfix manuel. FLF2V, T2V/I2V turbo et les LoRA utilisateur ne sont pas encore GPU-validés.
- Le générateur n’ajoute ni musique ni mixage audio ; les fondus actuels sont vidéo uniquement.
- Les workflows officiels évoluent. Le convertisseur échoue volontairement si un binding critique n’est plus identifiable.
- SQLite cible un seul conteneur et un seul worker ; une future exploitation multi-réplica nécessitera PostgreSQL et une file distribuée.
- Les licences CivitAI sont propres à chaque créateur et peuvent changer. Le champ `license_note` est informatif, pas un avis juridique.

Voir également [COMPATIBILITE.md](COMPATIBILITE.md), [DEPLOY_CLORE.md](DEPLOY_CLORE.md), [VERSIONS.md](VERSIONS.md) et [examples/sequence.json](examples/sequence.json).
