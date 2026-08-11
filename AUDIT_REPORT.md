# Rapport d’audit et de correction

Date : 2026-08-01.

## Arborescence finale

```text
wan-sequence-cloud/
├── .dockerignore
├── .env.example
├── .gitattributes
├── .gitignore
├── AUDIT_REPORT.md
├── COMPATIBILITE.md
├── DEPLOY_CLORE.md
├── Dockerfile
├── README.md
├── VERSIONS.md
├── docker-compose.yml
├── requirements.txt
├── app/
│   ├── __init__.py
│   ├── catalog.py
│   ├── comfy.py
│   ├── job_store.py
│   ├── main.py
│   ├── orchestrator.py
│   ├── schemas.py
│   ├── security.py
│   └── settings.py
├── config/
│   ├── base_models.json
│   ├── extra_model_paths.yaml
│   └── loras.json
├── examples/
│   └── sequence.json
├── scripts/
│   ├── build-and-push.sh
│   ├── download_base_models.py
│   ├── download_loras.py
│   ├── download_utils.py
│   ├── entrypoint.sh
│   └── prepare_workflows.py
└── tests/
    ├── test_backend.py
    └── test_contract.py
```

Les dossiers `__pycache__` préexistants ou générés par les tests sont ignorés par Git et Docker. Ils n’ont pas été supprimés, aucune option destructive `--clean` n’ayant été demandée.

## Fichiers créés

- `.gitattributes`, `.gitignore`, `AUDIT_REPORT.md`, `DEPLOY_CLORE.md`, `VERSIONS.md` ;
- `app/job_store.py`, `app/security.py`, `app/settings.py` ;
- `config/base_models.json` ;
- `scripts/build-and-push.sh`, `scripts/download_utils.py` ;
- `tests/test_backend.py`.

## Fichiers modifiés

- `.dockerignore`, `.env.example`, `Dockerfile`, `docker-compose.yml`, `requirements.txt`, `README.md` ;
- `app/catalog.py`, `app/comfy.py`, `app/main.py`, `app/orchestrator.py` ;
- `config/extra_model_paths.yaml`, `config/loras.json` ;
- `scripts/download_base_models.py`, `scripts/download_loras.py`, `scripts/entrypoint.sh`, `scripts/prepare_workflows.py`.

`app/schemas.py`, `app/__init__.py`, `COMPATIBILITE.md`, `examples/sequence.json` et `tests/test_contract.py` ont été conservés sans modification.

## Problèmes détectés dans l’ancien Dockerfile et le projet

- références flottantes `master` et `main` ;
- stockage persistant codé en dur sous `/data` ;
- installation mono-stage conservant Git et les outils de compilation ;
- aucune utilisation des caches BuildKit APT/PIP/Git ;
- possibilité de téléchargement automatique coûteux au démarrage ;
- processus ComfyUI en arrière-plan puis `exec uvicorn`, rendant la supervision et l’arrêt coordonné incomplets ;
- healthcheck limité à une route ne vérifiant pas l’état de préparation ;
- registre des jobs uniquement en mémoire ;
- aucune authentification des routes `/v1/*` ;
- scripts de téléchargement sans `--check`, `--force`, hash, manifeste ou reprise ;
- fichiers d’exclusion incomplets ;
- workflows et modèles référencés sur des branches flottantes ;
- image I2V téléchargée en FP16 alors que le template épinglé utilise FP8 ;
- absence de mode backend sans GPU et de documentation Clore.ai.

## Dockerfile corrigé

- cible unique `linux/amd64` ;
- build multi-stage : outils de compilation et Git absents du runtime ;
- cache BuildKit pour APT, PIP et Git ;
- PyTorch CUDA installé sans GPU pendant le build ;
- ComfyUI récupéré par fetch superficiel sur un commit ;
- trois petits workflows UI officiels récupérés à un commit précis ;
- aucun modèle, LoRA, checkpoint, média ou token dans l’image ;
- processus applicatifs exécutés sous `appuser` après initialisation des droits du volume par `gosu` ;
- seul le port 8000 est exposé ;
- healthcheck public sur `/health/live`.

## Versions et composants

- image exacte Linux AMD64 documentée : `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04@sha256:0bb88834d973ca1b450fcc2a05333c6fe45510bee289912a5391274c351c4a4d` ; elle n'a pas été tirée dans cet environnement ;
- CUDA : 12.4.1 ;
- PyTorch : 2.5.1 avec wheels CU124 ;
- ComfyUI : `2881e6161081439b1c3fb3b6c1f51b3d272da710` ;
- workflow templates : `1b3bdd46c945d54d893a3b43692d5963608fb7d4` ;
- custom nodes : aucun ;
- ENTRYPOINT exact : `["/app/scripts/entrypoint.sh"]` ;
- module FastAPI exact : `app.main:app` ;
- commande Uvicorn : `uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --workers 1` ;
- commande ComfyUI : `python /opt/ComfyUI/main.py --listen "$COMFYUI_HOST" --port "$COMFYUI_PORT" ...` ;
- ports : FastAPI `0.0.0.0:8000`, ComfyUI `127.0.0.1:8188`.

Les versions détaillées sont dans `VERSIONS.md`. Les dépendances directes de l’application et PyTorch sont épinglées. Certaines dépendances transitives déclarées par le `requirements.txt` du commit ComfyUI restent résolues par PIP au build ; un lock Linux produit après un premier build réussi renforcerait encore la reproductibilité.

## Variables d’environnement

| Variable | Défaut/usage |
|---|---|
| `DATA_ROOT` | `/workspace` |
| `PORT` | `8000` |
| `COMFYUI_HOST` | `127.0.0.1` |
| `COMFYUI_PORT` | `8188` |
| `GPU_CONCURRENCY` | `1`, seule valeur acceptée actuellement |
| `APP_TEST_MODE` | `0` |
| `API_TOKEN` | facultatif, recommandé |
| `REQUIRE_API_TOKEN` | `0`; avec `1`, exige `API_TOKEN` |
| `JOB_DATABASE_URL` | SQLite par défaut ; accepte actuellement `sqlite:///...` |
| `HF_TOKEN` | téléchargement HF explicite, jamais persisté |
| `CIVITAI_API_TOKEN` | téléchargement Civitai explicite, jamais persisté |
| `WORKFLOW_DIR` | `${DATA_ROOT}/cache/workflows` |
| `UI_WORKFLOW_DIR` | `/app/ui_workflows` |
| `LORA_CATALOG` | catalogue embarqué par défaut |
| `BASE_MODEL_CATALOG` | catalogue embarqué par défaut |
| `MAX_INPUT_DOWNLOAD_MB` | `50` |
| `COMFY_URL` | surcharge facultative du convertisseur de workflows |
| `IMAGE_REF` | obligatoire pour `build-and-push.sh` |
| `BUILD_CACHE_REF` | cache de registre facultatif |
| `MIN_BUILD_DISK_GB` | `100` |

## Chemins persistants

Tous sont dérivés de `DATA_ROOT` :

```text
models/
models/diffusion_models/
models/text_encoders/
models/vae/
models/checkpoints/
models/loras/
inputs/
inputs/comfy/
outputs/
outputs/comfy/
jobs/jobs.sqlite3
cache/
cache/workflows/
downloads/installed-files.json
```

`DATA_ROOT=/data` fonctionne également sans modification du code.

## Téléchargements

Pendant le build : image CUDA, paquets APT nécessaires, wheels Python, source ComfyUI épinglée et trois petits workflows JSON. Aucun modèle ni secret.

Après le build, uniquement sur commande explicite : modèles Wan, encodeur, VAE et LoRA. Les scripts vérifient taille/hash disponible, réutilisent les fichiers valides, reprennent `.part`, renomment atomiquement et écrivent un manifeste uniquement s’il change. `--force` impose un remplacement.

## Tailles et ressources estimées

- image sans modèles : estimation prudente 8–12 Go compressés et 20–30 Go décompressés ; non mesurée faute de build Docker ;
- espace temporaire de build : 100 Go minimum, 150 Go recommandé ;
- espace `/workspace` : 120 Go minimum pour les modèles listés et quelques résultats, 200 Go recommandé pour les caches et productions.

## Commandes

Build local sans publication :

```bash
docker buildx build --platform linux/amd64 --load -t wan-sequence:local .
```

Build et publication Clore.ai :

```bash
IMAGE_REF=registre/utilisateur/wan-sequence:1.0.0 ./scripts/build-and-push.sh
```

Lancement :

```bash
docker run --rm --gpus all -p 8000:8000 \
  -v /workspace:/workspace \
  -e DATA_ROOT=/workspace \
  -e API_TOKEN="$API_TOKEN" \
  -e REQUIRE_API_TOKEN=1 \
  wan-sequence:local
```

## Résultats exacts des contrôles

- `python -m compileall -q app scripts tests` : réussi ;
- `python -m unittest discover -s tests -v` : 8 tests exécutés, 8 réussis ;
- test mode : santé live/ready, authentification 401, job queued/running/completed, téléchargement factice et persistance après réouverture validés ;
- SQLite : transition `running` vers `interrupted` après réinitialisation validée ;
- téléchargement : taille/hash et manifeste inchangé validés ;
- `download_base_models.py --check` : code 1 attendu, six fichiers absents détectés, aucun téléchargement ;
- `download_loras.py --check` : code 1 attendu, onze fichiers absents détectés, aucun téléchargement ;
- JSON : tous les fichiers analysés avec succès ;
- OpenAPI : routes principales générées avec succès ;
- `docker compose config --quiet` : réussi ;
- audit : aucun secret réel et aucun fichier lourd supérieur à 1 Mo trouvé ;
- chemins : aucune donnée persistante codée sous `/data`, `/runpod-volume`, `/root` ou `/tmp`. `./data:/workspace` est seulement le chemin hôte Compose ; `/root/.cache/*` apparaît uniquement dans des mounts BuildKit non conservés dans l’image finale.

## Tests non exécutés

- build Docker : Docker Desktop/daemon Linux indisponible ; `docker buildx build --check` a échoué avant analyse avec pipe `dockerDesktopLinuxEngine` absent ;
- analyse Bash native : WSL/Bash inaccessible dans l’environnement ;
- téléchargement réel des modèles/LoRA : évité volontairement ;
- conversion des workflows contre un ComfyUI réellement lancé : exige l’image construite ;
- test GPU et génération Wan : GPU et modèles absents ;
- publication registre : non demandée et aucun registre défini.

Commandes ultérieures :

```bash
docker buildx build --check --platform linux/amd64 .
docker buildx build --platform linux/amd64 --load -t wan-sequence:local .
docker run --rm -p 8000:8000 -e APP_TEST_MODE=1 \
  -v "$PWD/test-data:/workspace" wan-sequence:local
python /app/scripts/download_base_models.py --check
python /app/scripts/download_loras.py --check
```

Une génération réelle requiert une machine NVIDIA, les modèles installés sous `DATA_ROOT`, idéalement au moins 48 Go de VRAM pour les workflows 14B natifs, et suffisamment de RAM/disque.

## Problèmes restant avant un déploiement réel

1. Le Dockerfile n’a pas encore été construit sur Linux AMD64.
2. Le digest de base a été vérifié dans les métadonnées officielles Docker Hub mais pas tiré par le daemon local.
3. Le convertisseur UI → API doit être validé contre le ComfyUI épinglé lors du premier build.
4. Les dépendances transitives ComfyUI ne disposent pas encore d’un lock Linux complet.
5. `generate_start_image` exige toujours un workflow image API et un checkpoint compatibles fournis séparément.
6. PostgreSQL n’est pas encore implémenté ; la cible actuelle est volontairement un seul conteneur SQLite.

## Éléments réutilisés sans modification

- `app/schemas.py` et tous ses modes vidéo ;
- `app/__init__.py` ;
- `COMPATIBILITE.md` ;
- `examples/sequence.json` ;
- `tests/test_contract.py` ;
- structure FastAPI/ComfyUI/FFmpeg existante ;
- compilation des workflows, bindings JSON et injection LoRA existants ;
- catalogue LoRA existant, enrichi seulement pour les métadonnées Wan et contrôles de fichiers ;
- dépendances applicatives compatibles déjà déclarées ; seules deux dépendances inutilisées/extras ont été retirées (`python-multipart`, extras Uvicorn).

## Opérations coûteuses évitées

- aucun modèle Wan, encodeur, VAE, checkpoint, GGUF ou LoRA téléchargé ;
- aucune vidéo générée et aucun test GPU exécuté ;
- aucune image Docker complète construite ;
- aucune publication vers un registre ;
- aucun clone complet : seules les références distantes ont été lues, et le Dockerfile utilisera un fetch Git superficiel ;
- aucune variante d’image construite ;
- aucune archive existante remplacée ;
- caches BuildKit APT/PIP/Git préparés pour réutiliser les couches système, PyTorch, ComfyUI et dépendances lors des builds ultérieurs ;
- les dépendances FastAPI de test ont été installées une seule fois dans `C:\tmp\wan-sequence-test-deps` et réutilisées pour la validation finale.

## Second correctif incrémental avant publication

### 1. Fichiers modifiés ou ajoutés

- Modifiés : `.env.example`, `Dockerfile`, `docker-compose.yml`, `README.md`, `DEPLOY_CLORE.md`, `VERSIONS.md`, `AUDIT_REPORT.md` ;
- modifiés : `app/job_store.py`, `app/main.py`, `app/orchestrator.py`, `app/schemas.py`, `app/settings.py` ;
- ajoutés : `app/limits.py`, `app/readiness.py`, `scripts/cleanup_jobs.py`, `tests/test_second_corrective.py` ;
- modifiés : `config/base_models.json`, `scripts/download_base_models.py`, `scripts/download_utils.py`, `tests/test_backend.py`.

### 2. Digest Docker AMD64

Le digest documenté et utilisé est `sha256:0bb88834d973ca1b450fcc2a05333c6fe45510bee289912a5391274c351c4a4d`. L'image n'a pas été tirée dans cet environnement.

### 3. Permissions Git

Le dossier n'étant pas un dépôt, un dépôt Git local vide a été initialisé sans remote ni commit. `scripts/build-and-push.sh` et `scripts/entrypoint.sh` sont indexés en mode `100755`; leurs fins de ligne ont été vérifiées en LF (`0` CRLF). Le fallback `bash scripts/build-and-push.sh` est documenté.

### 4. Profils de téléchargement

- `t2v` : `wan22_t2v_high`, `wan22_t2v_low`, `umt5_xxl`, `wan21_vae` ;
- `i2v` et `flf2v` : `wan22_i2v_high`, `wan22_i2v_low`, `umt5_xxl`, `wan21_vae` ;
- `all` : union dédupliquée des profils T2V et I2V.

Sans `--profile`, `--ids`, `--list`, `--estimate` ou `--check`, le script sort avec l'aide et ne télécharge rien. Les six entrées disposent d'une taille exacte, d'un SHA-256 LFS officiel, d'un identifiant Xet, d'une révision et d'un chemin épinglés.

### 5. Exemple `--estimate`

`python scripts/download_base_models.py --profile t2v --estimate` a indiqué, sans réseau ni création du dossier de modèles : `expected=33.13 GiB`, `present_valid=0.00 B`, `remaining=33.13 GiB`, avec le détail des quatre fichiers.

### 6. Variables de limitation

`MAX_SHOTS_PER_JOB=8`, `MAX_PENDING_JOBS=2`, `MAX_TOTAL_FRAMES=968`, `MAX_OUTPUT_DISK_GB=50`, `MIN_FREE_DISK_GB=10`, `MAX_JOB_COST_UNITS=8`.

### 7. Formule d'estimation

Référence : `832 × 480 × 121 × 20`. Pour chaque plan : `width × height × frames × steps / référence`; le job est la somme des plans. Cette unité estime du calcul relatif, jamais un prix monétaire. Elle est enregistrée dans la réponse, SQLite, les logs et le statut.

### 8. Règles d'assemblage

Tous les plans doivent avoir largeur, hauteur et FPS identiques. Le nombre de frames peut varier. Un `crossfade` doit être strictement plus court que chaque plan concerné. Ces règles sont contrôlées avant l'insertion SQLite, le dossier de sortie et la réservation GPU.

### 9. `READINESS_PROFILE`

`backend` (défaut) vérifie séparément API, SQLite et répertoires et peut être prêt sans ComfyUI ni modèles. `t2v`, `i2v`, `flf2v` et `all` exigent en plus ComfyUI, les workflows API et les fichiers du profil. La vérification modèle lit présence, taille exacte et manifeste, sans hash multi-Go ni chargement VRAM.

### 10. Protection SSRF

Les protocoles sont limités à HTTP/HTTPS, les redirections sont suivies manuellement (maximum 5), et chaque destination est résolue et rejetée si elle pointe vers loopback, privé ou link-local. Un changement DNS pour un hôte revu est rejeté. Le type MIME doit être `image/*`; `Content-Length` et le flux restent limités. `follow_redirects=True` n'est plus utilisé.

### 11. Tests exécutés

- compilation Python ;
- 20 tests unitaires : backend/test mode, persistance, manifeste, digest, profils/estimation hors ligne, coût, file, disque simulé, cohérence vidéo, readiness hors ligne, redirections HTTP simulées et rétention sèche ;
- syntaxe des deux scripts Bash avec Git Bash ;
- analyse des trois JSON ;
- `docker compose config --quiet` ;
- audit statique des anciens digests, redirections automatiques et seuils minimaux du catalogue ;
- permissions Git et fins de ligne LF.

### 12. Tests non exécutés

Aucun téléchargement de modèle/LoRA, build ou pull Docker, publication, test GPU, chargement VRAM, génération Wan ou assemblage vidéo réel. Ils nécessitent respectivement réseau et dizaines de Gio, daemon Docker Linux, registre, GPU NVIDIA et modèles installés.

### 13. Blocages restants avant GitHub

Le dépôt local n'a encore ni commit ni remote GitHub et les fichiers hors des deux scripts ne sont pas encore ajoutés à l'index. Le build Linux AMD64, la conversion réelle des workflows avec le ComfyUI épinglé et une inférence GPU restent volontairement non validés.

## Éléments réutilisés sans modification

- architecture FastAPI → ComfyUI → FFmpeg, SQLite et mode test ;
- modes vidéo, validation Wan/LoRA, catalogue LoRA, versions PyTorch/CUDA et commits ComfyUI/workflows ;
- scripts d'entrypoint/build (contenu inchangé), dépendances déclarées, workflows et exemples ;
- fichiers UMT5/VAE communs sont désormais explicitement réutilisables entre profils lorsqu'ils sont valides.

## Opérations coûteuses évitées

- aucun modèle, LoRA ou média téléchargé et aucun hash de fichier multi-Go exécuté localement ;
- aucune image Docker tirée ou construite, aucun registre contacté, aucune publication ;
- aucun GPU réservé, aucune vidéo générée, aucune normalisation ou concaténation FFmpeg réelle ;
- les couches stables Docker système, PyTorch, ComfyUI et dépendances restent séparées et réutilisables via BuildKit ;
- seules les pages/pointeurs LFS officiels, de quelques octets, ont été consultés pour les tailles et hashes.

## Correctif runtime Clore.ai : PATH, permissions et sous-graphes Wan

### 1. Causes racines confirmées

1. `scripts/entrypoint.sh` appelait `python` et `uvicorn` par leur nom. Le SSH
   Autoinstall de Clore peut remplacer l'environnement et retirer
   `/opt/venv/bin` du `PATH`, alors que les deux exécutables existent bien dans
   ce virtualenv.
2. L'image ne préparait pas `/opt/ComfyUI/user` et `/opt/ComfyUI/temp` pour
   `appuser`. Les deux `PermissionError` observées correspondent exactement à
   ces chemins runtime.
3. Le convertisseur ne parcourait que `workflow["nodes"]`. Au commit workflow
   épinglé, T2V possède quatre nœuds principaux et place les nœuds de prompt,
   sampling, dimensions et création vidéo dans `definitions.subgraphs`. I2V a
   la même forme générale, avec son `LoadImage` dans le graphe principal. La
   validation stricte échouait donc après le démarrage réussi de ComfyUI.

### 2. Hypothèses examinées puis rejetées

- Les tentatives `curl: (7)` transitoires font partie de la boucle d'attente et
  ne sont pas un incident si `/system_stats` finit par répondre.
- Les avertissements PyTorch/CUDA ne causent pas la panne constatée ; aucune
  version CUDA, PyTorch ou image de base n'a été modifiée.
- Les commits ne sont pas flottants : ComfyUI reste épinglé à
  `2881e6161081439b1c3fb3b6c1f51b3d272da710` et les templates à
  `1b3bdd46c945d54d893a3b43692d5963608fb7d4`.
- FLF2V n'a pas le bug de sous-graphe à ce commit : ses 40 nœuds sont au niveau
  principal. Le convertisseur conserve la branche normale et ignore la branche
  Lightning marquée `mode: 4`.
- Il n'est pas nécessaire d'installer un second Python système ni de transférer
  récursivement la propriété de tout `/opt/ComfyUI` à `appuser`.

### 3. Fichiers modifiés ou ajoutés

- modifiés : `Dockerfile`, `scripts/entrypoint.sh`,
  `scripts/prepare_workflows.py`, `README.md`, `DEPLOY_CLORE.md`, `VERSIONS.md`,
  `AUDIT_REPORT.md` ;
- ajoutés : `scripts/smoke_runtime.sh`, `tests/test_workflow_conversion.py` et
  `tests/fixtures/workflow_templates_1b3bdd46/{README.md,video_wan2_2_14B_t2v.json,video_wan2_2_14B_i2v.json,video_wan2_2_14B_flf2v.json}`.

### 4. Détail des modifications

- L'entrypoint exporte `/opt/venv/bin`, vérifie les deux exécutables puis lance
  ComfyUI et le convertisseur avec `/opt/venv/bin/python`, et FastAPI avec
  `/opt/venv/bin/uvicorn`. ComfyUI reste lié à `127.0.0.1:8188` et Uvicorn à
  `0.0.0.0:8000`.
- Le Dockerfile crée uniquement `/opt/ComfyUI/user` et
  `/opt/ComfyUI/temp` comme chemins ComfyUI appartenant à `appuser`.
  L'entrypoint les recrée de façon idempotente lors d'un lancement initial en
  root, puis vérifie leurs droits après `gosu`. Les entrées, sorties, modèles,
  jobs, cache et workflows restent sous le volume `/workspace` déjà géré.
- `prepare_workflows.py` normalise les deux formats de liens, développe les
  sous-graphes sur un niveau, espace les identifiants par instance et reconnecte
  leurs frontières d'entrée/sortie au graphe principal. Il sérialise ensuite les
  nœuds connus par le `/object_info` du ComfyUI démarré.
- La validation reste bloquante. Les bindings communs exigés sont prompts,
  seed, dimensions, frames, fps, steps, cfg, préfixe de sortie et cibles modèles
  high/low ; I2V exige aussi `start_image`, FLF2V `start_image` et `end_image`.
- Les sorties observées dans les tests sont notamment : T2V
  `positive=/subgraph:128:89/inputs/text`,
  `negative=/subgraph:128:72/inputs/text`,
  `seed=/subgraph:128:81/inputs/noise_seed`, dimensions/frames sur le nœud 74 et
  préfixe sur le nœud principal 80 ; I2V utilise les nœuds 93, 89, 86, 98 et le
  `LoadImage` principal 97 ; FLF2V utilise les nœuds principaux 90, 78, 84, 81,
  80, 89 et 83.
- Le smoke test runtime vérifie les exécutables sous un `PATH` minimal, les
  droits effectifs de `appuser`, les trois JSON et leurs `_bindings`, puis
  `/health/live` et la structure de `/health/ready`. Il ne masque pas un `503`
  de readiness dû à l'absence légitime des modèles.

### 5. Tests ajoutés

- empreinte canonique des trois fixtures et concordance du commit Dockerfile ;
- conversion T2V, I2V et FLF2V, classes/champs réellement ciblés, intégrité de
  tous les liens API et exclusion de la branche FLF2V désactivée ;
- génération bout en bout des trois `.api.json` temporaires avec `_bindings` ;
- régression prouvant que l'absence d'un binding reste une erreur ;
- contrôles statiques PATH, adresses d'écoute, permissions minimales, routes de
  santé et couverture du smoke test runtime.

### 6. Commandes exécutées

```text
python -m unittest tests.test_workflow_conversion -v
python -m unittest discover -s tests -v
python -m compileall -q app scripts
"C:\Program Files\Git\bin\bash.exe" -n scripts/entrypoint.sh scripts/build-and-push.sh scripts/smoke_runtime.sh
docker compose config --quiet
```

Des audits locaux complémentaires ont vérifié les pointeurs de bindings, les
liens entre nœuds, l'absence de CRLF dans les trois scripts Bash et les usages
de variables de secrets. Aucun secret concret n'a été trouvé.

### 7. Résultats complets

- suite ciblée : 8 tests, tous réussis ;
- suite complète finale : 28 tests, tous réussis en 0,971 s ;
- compilation Python : réussie ;
- syntaxe Bash : réussie ;
- validation Compose : réussie. Docker a averti que
  `C:\Users\celio\.docker\config.json` n'était pas lisible dans le sandbox,
  sans erreur de configuration Compose ;
- les trois prompts de test contiennent respectivement 31 nœuds T2V, 32 nœuds
  I2V et 16 nœuds FLF2V, avec tous leurs liens résolus.

### 8. Ce qui a réellement été testé sans GPU

Les trois sources exactes du commit épinglé ont été conservées comme fixtures
locales puis converties avec un `object_info` déterministe couvrant les nœuds
utilisés. Les fichiers API générés ont été relus comme JSON et leurs bindings et
liens vérifiés. FastAPI, `/health/live`, `/health/ready`, SQLite et le mode test
étaient déjà couverts par la suite complète. Aucun modèle, aucune VRAM et aucun
service externe ne sont nécessaires à ces tests.

Le smoke test embarqué n'a pas été exécuté, puisqu'il exige une image construite
et un conteneur réellement démarré. Il pourra l'être avec :

```bash
docker exec CONTAINER /app/scripts/smoke_runtime.sh
```

### 9. Validation restante sur RTX 4090 / serveur GPU

Après autorisation de construire une unique image Linux AMD64 : démarrer via le
SSH Autoinstall Clore avec `/app/scripts/entrypoint.sh`, confirmer dans les logs
ComfyUI puis la conversion puis Uvicorn, exécuter le smoke test ci-dessus et
vérifier `GET /health/live`. Avec les modèles installés explicitement, tester
ensuite le profil readiness voulu puis une génération courte par mode. Cette
dernière étape nécessite la RTX 4090, environ 24 Gio de VRAM, les modèles Wan et
un espace disque persistant suffisant ; l'offload peut rester nécessaire.

### 10. Risques résiduels

- Le développement générique est volontairement limité à un niveau de
  sous-graphe ; les templates épinglés respectent cette contrainte.
- La compatibilité avec le `/object_info` du ComfyUI réellement construit reste
  à confirmer par le premier démarrage, même si les classes des fixtures sont
  celles du workflow officiel épinglé.
- De futurs custom nodes pourraient introduire d'autres chemins inscriptibles ;
  aucun custom node n'est actuellement installé.
- La conversion réussie ne garantit pas qu'un workflow 14B complet tienne dans
  24 Gio sans offload, ni qu'un modèle/LoRA absent devienne prêt.

## Éléments réutilisés sans modification

- architecture FastAPI → ComfyUI → FFmpeg, SQLite, orchestration et schémas de
  payload ;
- logique existante de `/health/live` et `/health/ready` ;
- commits ComfyUI/workflow templates, versions CUDA/PyTorch et image de base ;
- catalogues modèles/LoRA, scripts de téléchargement et manifestes ;
- dépendances Python et système déjà déclarées ; aucune installation ajoutée ;
- branche active FLF2V et comportement strict du convertisseur.

## Opérations coûteuses évitées

- aucune image Docker construite, reconstruite, tirée ou publiée ;
- aucun modèle Wan, LoRA, checkpoint ou fichier multi-Gio téléchargé ou hashé ;
- aucune génération vidéo, inférence, allocation GPU ou réservation Clore ;
- aucune mise à niveau CUDA/PyTorch et aucun clone ComfyUI supplémentaire ;
- seules les petites sources JSON officielles épinglées ont servi aux fixtures ;
- les couches Docker stables système, PyTorch, ComfyUI et dépendances restent
  inchangées et donc réutilisables par BuildKit lors d'un futur build autorisé.

## Correctif sampling, injection LoRA et cache convertisseur

### Causes

1. `steps` et `cfg` ne contenaient qu'un pointeur, choisi sur le premier
   `KSamplerAdvanced`. La seconde phase conservait donc sa valeur ou son lien
   original, et le seuil `end_at_step`/`start_at_step` n'était pas recalculé.
2. La cible modèle était le premier consommateur du `UNETLoader`. Dans T2V et
   I2V, il s'agit du LoRA Lightning officiel situé uniquement sur le côté
   `on_true` d'un `ComfySwitchNode` désactivé par défaut. Une LoRA utilisateur
   injectée à cet endroit était contournée par le chemin raw actif.
3. `reusable()` ne comparait que le hash du workflow source. Une ancienne
   conversion restait donc considérée valide après une modification du
   convertisseur.

### Correction

- Pour T2V/I2V, les bindings suivent les switches statiques et modifient les
  primitives partagées de la branche active : total de steps, séparation et
  cfg. Pour FLF2V, ils ciblent explicitement les deux samplers. Le split conserve
  son ratio source (`10/20`, donc `16/33` par troncature) et la seconde phase se
  termine au nouveau total.
- Les cibles LoRA sont déterminées en remontant les dépendances du champ `model`
  de chaque `ModelSamplingSD3` jusqu'au loader high/low. Elles se trouvent ainsi
  après le switch raw/Lightning pour T2V/I2V et restent les nœuds 73/74 pour
  FLF2V.
- Le schéma convertisseur passe à `2`. Il est écrit sous
  `_converter_schema_version`, contrôlé par `reusable()`, retiré avant envoi à
  ComfyUI et vérifié par le smoke test runtime.

### Régression ajoutée

- application réelle de `steps=33` et `cfg=7.5` aux trois prompts, avec contrôle
  des deux samplers, du split à 16 et de la fin à 33 ;
- appel réel à `inject_loras()` avec deux fausses LoRA et vérification que les
  chemins high et low consommés par les samplers traversent les nœuds injectés ;
- refus d'un cache sans version ou avec une ancienne version, acceptation de la
  version courante et présence de la version dans chaque JSON généré.

### Fichiers et validation

Fichiers modifiés par ce correctif : `scripts/prepare_workflows.py`,
`app/comfy.py`, `tests/test_workflow_conversion.py`,
`scripts/smoke_runtime.sh`, `VERSIONS.md` et `AUDIT_REPORT.md`.

La suite complète conserve les 28 tests antérieurs et atteint 32 tests : tous
ont réussi en 0,959 s. `compileall`, la syntaxe des trois scripts Bash et
`docker compose config --quiet` ont également réussi. Le warning d'accès au
fichier Docker utilisateur du poste Windows n'a pas invalidé la configuration.

Aucun build, push, téléchargement de modèle/LoRA ou test GPU n'a été exécuté.
Sur RTX 4090, il reste à démarrer l'image autorisée ultérieurement, laisser
`prepare_workflows.py` régénérer les caches avec le schéma courant, exécuter
`/app/scripts/smoke_runtime.sh`, puis réaliser une courte génération T2V, I2V et
FLF2V avec des valeurs non par défaut et, séparément, une LoRA high/low réelle.

## Intégration permanente des hotfixes runtime validés sur RTX 4090

### Bugs corrigés

1. Les trois templates sérialisaient `SaveVideo` avec une valeur de format non
   compatible avec le ComfyUI réellement déployé. La préparation force désormais
   chaque `SaveVideo` à `format=mp4` et `codec=auto`, sans dépendre de l'ID 80,
   tout en conservant `filename_prefix` et son binding.
2. T2V et I2V contenaient chacun deux LoRA LightX2V optionnelles. ComfyUI validant
   aussi la branche non sélectionnée, leur absence sous `/workspace/models/loras`
   invalidait le prompt. Le convertisseur reconnaît maintenant la topologie
   `UNETLoader → LoraLoaderModelOnly(lightx2v) → ComfySwitchNode →
   ModelSamplingSD3`, relie les deux côtés du switch au loader raw et retire le
   nœud Lightning devenu orphelin. FLF2V ne présente pas cette topologie active.
3. `queue_and_wait()` masquait le JSON d'erreur de `/prompt`. Une
   `ComfyPromptError` conserve maintenant le status HTTP et les champs `error` et
   `node_errors`, avec fallback texte borné et masquage des secrets usuels.
4. L'entrypoint créait les feuilles comme `models/loras`, mais pas forcément les
   parents `models`, `inputs` et `outputs` avec des droits appuser. Tous les
   `required_dirs` sont maintenant créés en ordre avec `install -d -m 775`, sans
   parcours récursif des modèles ni destruction des données existantes.
5. Le downloader pouvait finir par un `FileNotFoundError` trompeur lors de la
   promotion du `.part`. Il distingue désormais erreurs HTTP, interruption,
   longueur incomplète, reprise incohérente, `.part` absent ou vide, et conserve
   toujours une cible existante jusqu'à la promotion atomique réussie.

Le schéma convertisseur passe de 2 à 3 afin de régénérer automatiquement les
anciens prompts qui contiennent encore les branches Lightning ou l'ancien
`SaveVideo`.

### Fichiers modifiés ou ajoutés

- `scripts/prepare_workflows.py`, `scripts/entrypoint.sh`,
  `scripts/download_utils.py`, `scripts/download_base_models.py`,
  `scripts/smoke_runtime.sh` ;
- `app/comfy.py`, `Dockerfile` ;
- `tests/test_workflow_conversion.py` et nouveau
  `tests/test_runtime_hotfixes.py` ;
- `README.md`, `VERSIONS.md`, `DEPLOY_CLORE.md`, `AUDIT_REPORT.md`.

### Validation hors GPU

La suite complète finale a exécuté 44 tests en 1,142 s, tous réussis. Elle couvre les
28 tests antérieurs ainsi que : MP4/codec des trois workflows, suppression
sémantique Lightning T2V/I2V, chemins high/low, bindings, double phase
steps/cfg, injection LoRA utilisateur, erreur ComfyUI JSON et texte, création
filesystem idempotente, téléchargement normal, reprise 206, serveur ignorant
Range, `.part` disparu, HTTP 503, interruption, conservation d'une cible valide
et absence de réseau/réécriture pour modèle+manifeste valides.

Ont également réussi : `python -m compileall -q app scripts tests`, la syntaxe
des trois scripts Bash et `docker compose config --quiet`. Le warning Windows
sur la lecture de `.docker/config.json` n'a pas invalidé Compose.

Le smoke runtime vérifie désormais le schéma 4, les bindings, `SaveVideo
mp4/auto` et la recette LightX2V optionnelle. Aucune image n'a été
construite ou poussée et aucun gros fichier n'a été téléchargé.

### Validation restante

Sur la prochaine RTX 4090 : confirmer la régénération du cache en schéma 4,
exécuter `/app/scripts/smoke_runtime.sh`, vérifier `/health/ready`, puis lancer
un T2V court. I2V/FLF2V et une LoRA utilisateur réelle restent également à
valider en inférence. Le comportement réel d'un serveur HTTP interrompant une
reprise multi-Gio reste à observer malgré sa couverture simulée.

Commande recommandée après autorisation, avec un tag neuf qui ne remplace pas
l'image fonctionnelle :

```bash
IMAGE_REF=ghcr.io/OWNER/wan-sequence-cloud:v2-runtime-hotfix-20260809 \
  bash scripts/build-and-push.sh
```

## Troisième patch incrémental — transport OCI des modèles

Les modèles de base ne sont plus destinés à une layer monolithique. `prepare_bundled_models.py` s'exécute sur l'hôte avant BuildKit, valide les sources épinglées, produit des chunks de 4 GiB et un manifest version 1, puis génère `Dockerfile.bundled` avec exactement un `COPY --link` par chunk. `.bundled-models/work` n'est jamais inclus dans le contexte ; le cache registry `mode=max` ne peut donc exporter que les layers de chunks.

Au runtime, `materialize_bundled_models.py` vérifie chunks, espace libre et SHA-256, écrit vers `.part`, fsync puis promeut atomiquement sous `/workspace/models`. Il renseigne `installed-files.json` avec `installed_from=bundled-image`. La readiness vérifie désormais également le SHA-256 final.

Les LoRA utilisent une source et une URL explicites du catalogue. Les téléchargements HF/Civitai sont résumables, verrouillés par fichier et validés avant injection. Aucun token n'est persisté. LightX2V est téléchargée de cette manière uniquement pour un shot turbo.

## Quatrième patch incrémental — profils produit complets

L’audit de `_generate_keyframe()`, de la documentation et des fichiers présents confirme que `sd15`, `illustrious`, `pony` et `anima` ne sont que des valeurs de schéma : aucun `image_*.api.json`, checkpoint, URL épinglée, taille ou SHA-256 de base n’est fourni. Les déclarer prêts aurait sélectionné arbitrairement des poids et une topologie non audités.

Le catalogue définit désormais `all-video`, les capacités requises par profil et l’état explicite de chaque engine image. `all-video` est l’union dédupliquée T2V/I2V/FLF2V. `all` reste l’union maximale effectivement supportée et est donc identique à `all-video` dans cette révision. La readiness détaille les trois familles vidéo et `image_generation`; le profil `all` ne peut pas devenir prêt tant que l’image ne l’est pas. `_generate_keyframe()` refuse l’engine avant tout téléchargement de LoRA et vérifie, pour un futur engine activé, son manifeste puis l’identité du checkpoint référencé par le workflow.

Le bundler et le matérialiseur n’ont pas été dupliqués : leurs chemins relatifs génériques prennent déjà en charge `checkpoints/...`, avec chunks vérifiés, contrôle d’espace, écriture `.part`, SHA-256 final et promotion atomique. Les tests emploient un petit faux checkpoint pour valider ce chemin sans téléchargement, build, registre ou GPU.

Validation hors GPU du patch : `74 passed, 18 subtests passed in 2.71s`; compilation de `app`, `scripts` et `tests` réussie ; 7 fichiers JSON valides ; syntaxe des trois scripts Bash réussie avec Git Bash ; `docker compose config --quiet` réussi malgré le warning Windows déjà connu sur `.docker/config.json`. Les profils `t2v`, `all-video` et `all` ont été générés statiquement avec de petites fixtures et zéro appel réseau. Aucun vrai modèle, build Docker, push ou GPU n’a été utilisé.

## Cinquième patch incrémental — FLUX.1-schnell

La capacité image est désormais fondée exclusivement sur `flux_schnell` et le workflow natif à loaders séparés : diffusion model, `DualCLIPLoader` CLIP-L/T5XXL, AE, encodage FLUX, latent SD3, sampling Schnell, décodage et `SaveImage`. Les quatre pointeurs officiels ont été épinglés avec taille et SHA-256 ; Wan UMT5 reste distinct du T5XXL FLUX.

`POST /v1/images` crée un image job asynchrone, les routes de statut et output renvoient le job puis un PNG. Une référence interne opaque `img_<uuid>` peut alimenter `start_image.image_id` sans exposer de chemin. `generate_start_image` appelle le même générateur avant Wan I2V. `start_image` et `generate_start_image` sont mutuellement exclusifs ; une image fournie ne déclenche jamais FLUX.

Le profil `image`/`flux-schnell` contient 29 257 890 132 octets et `all` 93 424 945 275 octets. Le bundler et le matérialiseur génériques sont réutilisés sans conversion des poids. LightX2V et les LoRA utilisateur restent dynamiques.

Validation hors GPU du cinquième patch : `84 passed, 18 subtests passed in 2.58s` ; compilation de `app`, `scripts` et `tests`, validation de tous les JSON, syntaxe des trois scripts Bash et `docker compose config --quiet` réussies. L'estimation locale du profil `image` retrouve 27,25 Gio à matérialiser et ne fait aucun accès réseau. Aucun modèle, LoRA ou workflow n'a été téléchargé ; aucune image Docker n'a été construite ou poussée et aucun GPU n'a été utilisé.

À la fin de ce cinquième patch, la validation RTX 4090 restait à faire. Elle a depuis été réalisée sur la release 0.5.0 après hotfix pour FLUX standalone, la réutilisation par Wan I2V et le chemin `generate_start_image`; le sixième patch ci-dessous distingue ces résultats du cold-start encore non validé sans intervention.

## Sixième patch incrémental — release cold-start 0.5.1

La release 0.5.0 bundlée avait bien produit les 29 chunks attendus, mais `COPY --link --chmod=0444` vers des destinations imbriquées avait aussi laissé les parents créés implicitement sans bit `x`. Le générateur conserve désormais chaque `COPY --link --chmod=0444`, puis ajoute une unique couche de métadonnées qui fixe seulement les répertoires sous `/opt/wan-model-parts` à `0555` et vérifie que les fichiers n'ont aucun bit d'écriture. Les chunks ne sont ni fusionnés ni transformés en shards safetensors.

Le second défaut provenait de `uvicorn app.main:app` lancé relativement au CWD. L'entrypoint résout désormais son propre répertoire, fixe `APP_ROOT`, exécute `cd "$APP_ROOT"`, utilise des chemins de scripts absolus et passe `--app-dir "$APP_ROOT"` à Uvicorn. Un preflight d'entrypoint exécuté depuis un répertoire temporaire différent de `/app` a importé `app.main` et confirmé `READINESS_PROFILE=all`, `REQUIRE_API_TOKEN=1` et `API_TOKEN=set` sans imprimer la valeur synthétique du token.

`POST /v1/images/upload` reçoit maintenant PNG/JPEG en multipart authentifié. Le flux est limité en octets, décodé avec Pillow, protégé par une limite de pixels/decompression bomb, transposé EXIF, réencodé en PNG, écrit via temporaire puis `os.replace`, et enregistré comme image job `completed`. Le filename et le Content-Type client sont ignorés pour la confiance ; le statut public n'expose pas le chemin serveur. L'image fonctionne ensuite comme `start_image` I2V et comme `start_image`/`end_image` FLF2V. `cleanup_jobs.py` sait supprimer ses chemins contrôlés. `python-multipart==0.0.22` est la seule dépendance ajoutée ; Pillow déjà installé par ComfyUI est réutilisé.

L'incohérence LoRA standalone est résolue par l'option A : `ImageGenerationRequest` accepte `loras`, mais uniquement avec cible `auto`/`keyframe`; la compatibilité `flux_schnell` est validée avant ComfyUI, puis le mécanisme dynamique existant télécharge et injecte réellement la LoRA. Une famille incompatible reste refusée.

Le preflight `bash scripts/preflight-release.sh` a réussi : compilation Python, imports, 8 JSON applicatifs, syntaxe Bash, `93 passed, 18 subtests passed in 6.53s`, faux profil `all` avec 10 modèles et 53 chunks, première matérialisation `materialized`, seconde `already_valid`, et Compose valide malgré l'avertissement Windows connu sur `.docker/config.json`. Le daemon Docker local était indisponible : la mini-image et les contrôles Linux réels sous UID 10001 n'ont pas été exécutés. Aucun modèle lourd, build de production, push GHCR ou GPU n'a été utilisé.

Les validations GPU réelles communiquées pour la 0.5.0 après hotfix sont FLUX standalone, Wan T2V standard, Wan I2V avec `image_id` et I2V avec `generate_start_image` sur RTX 4090. FLF2V, LightX2V T2V/I2V, LoRA utilisateur et cold-start sans hotfix ne doivent toujours pas être annoncés comme GPU-validés.

Commande future non exécutée :

```bash
BUNDLED_MODEL_PROFILE="all" \
IMAGE_REF="ghcr.io/leselyo/wan-sequence-cloud-v2:0.5.1-all-flux" \
MIN_BUILD_DISK_GB="100" \
bash scripts/build-and-push.sh
```

### RELEASE BUILD GATE

```text
Permission regression:
PASS

Bundle traversal as appuser:
FAIL — not yet proven in a Linux image because Docker is unavailable

Bundle immutable files:
PASS

Mini materialization:
PASS

Second materialization already_valid:
PASS

Entrypoint from /root:
FAIL — root/gosu container acceptance test not run because Docker is unavailable

Python app import independent of cwd:
PASS

Environment propagation:
PASS

Bearer auth:
PASS

Image upload security tests:
PASS

All previous tests:
PASS

Generated bundled Dockerfile:
PASS

Mini Docker image:
NOT RUN — Docker unavailable

Large model download:
NOT RUN intentionally

Large production Docker build:
NOT RUN intentionally

GHCR push:
NOT RUN intentionally

Recommendation to perform expensive production build:
NO
```
