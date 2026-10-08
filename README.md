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

## Montage FFmpeg : plan « jump cut » (`mode: "jumpcut"`)

Un plan peut être un montage de photos fixes coupées par FFmpeg **sur le serveur**, sans GPU ni workflow ComfyUI, et se concatène ensuite avec les autres plans (Wan, Animate, VACE) dans le même job. Il n'existe volontairement **aucune route qui reçoit des arguments FFmpeg bruts** : une ligne de commande libre permettrait de lire n'importe quel fichier du service ou d'appeler n'importe quelle URL. L'API expose des options typées, que `app/montage.py` traduit en commandes FFmpeg (même code que `scripts/trend_philosopher.py` : `jumpcut_lengths`, `build_jumpcut`, `shake_filter`).

```bash
curl -H "Authorization: Bearer $API_TOKEN" http://localhost:8000/v1/montage        # choix possibles, valeurs par défaut, FFmpeg du serveur
curl -X POST http://localhost:8000/v1/jobs -H "Authorization: Bearer $API_TOKEN" \
  -H 'Content-Type: application/json' --data-binary @examples/sequence-jumpcut.json
```

Les images viennent de `/v1/images` (Krea2, FLUX), de `/v1/images/upload`, d'un `path` du dossier d'entrée ou d'une URL publique, exactement comme pour les autres plans. Options de `jumpcut` (toutes facultatives sauf `images`) :

| Option | Défaut | Rôle |
|---|---|---|
| `images` | requis, 2 à 40 | les photos, dans l'ordre d'affichage |
| `ratios` | aucun (automatique) | un nombre par image : `[1, 1, 2, 1]` sur 5 s = 1 s, 1 s, 2 s, 1 s. Sans `ratios` : images mélangées, chacune entre `cut_min_seconds` et `cut_max_seconds` |
| `transition` | `cut` | enchaînement d'une image à l'autre : `cut`, `shake` (coupe + secousse), `wipe_left`, `wipe_right` (volet), `swing` (le volet alterne gauche / droite) |
| `transition_seconds` | 0,10 | durée d'un volet (limitée à la moitié de la plus courte image) |
| `constant_shake` | aucun | tremblement de caméra constant : `jitter`, `roll`, `handheld`, `sway`, `rock` (lent, lisse, cadrage inchangé), `drift` (aussi lent que `rock` mais la direction et l'angle changent sans cesse) |
| `constant_shake_amount` | 1,0 | intensité du tremblement (1,0 = le préréglage) |
| `cut_min_seconds` / `cut_max_seconds` | 0,2 / 0,4 | durée d'une image en mode automatique |

La durée du plan est `frames / fps` (17 à 241 images), `width` / `height` / `fps` doivent être les mêmes que pour les autres plans du job, et une image ne peut pas rester moins de 0,05 s (le plan est refusé en 422 avec le calcul). `seed` fixe l'ordre mélangé du mode automatique. Le coût estimé est de 0,02 unité par plan, le montage n'utilisant que le CPU. Quand un job contient un plan `jumpcut`, la concaténation ré-encode la vidéo (sans audio) au lieu de copier les flux, pour éviter les accrocs entre deux encodeurs différents.

**FFmpeg du serveur.** L'image Docker installe un FFmpeg statique récent (branche 8.1, build BtbN) dans `/opt/ffmpeg` et le place en tête du `PATH` (`FFMPEG_BIN` / `FFPROBE_BIN` le désignent explicitement) ; le paquet Ubuntu 22.04 (4.4.2) ne sert plus que de repli. Le build vérifie que les filtres `xfade`, `fillborders`, `perspective`, `rotate`, `pad`, `overlay`, `ass` et l'encodeur `libx264` existent. `GET /health/ready` rapporte `capabilities.montage` (version et filtres manquants, jamais requis par un profil), et un plan dont les options demandent un filtre absent est refusé à l'admission (422) plutôt que d'échouer en plein job. `--build-arg FFMPEG_SHA256=…` (et un `FFMPEG_URL` daté) rend l'image reproductible ; sans cela, le build affiche seulement l'empreinte.

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

## Wan 2.2 Animate : clips pilotes, mode Move et tendances chat

Validé sur RTX 4090 (2026-10-02). Le modèle Animate transfère la pose d'une personne filmée (la vidéo « pilote », `driving_video`) sur le personnage de `start_image`.

- **`animate_mix`** : le personnage remplace la personne ; le décor de la vidéo pilote est conservé (masque SAM2 + fond noirci). SAM2 reçoit des points de deux sortes, comme dans l'éditeur de points de ComfyUI : **vert** (positif) sur la personne à remplacer, **rouge** (négatif) sur ce qui doit rester hors du masque (fond, l'autre combattant). Le point vert est calculé par détection YOLO de la personne sur la première image ; `subject_point: [x, y]` (0 à 1, première image de la vidéo pilote) le remplace quand la « meilleure personne » n'est pas la bonne, par exemple dans un combat à deux. `exclude_points: [[x, y], ...]` ajoute des points rouges. Le gabarit officiel ne relie que la sortie positive de `PointsEditor` à `Sam2Segmentation` (`coordinates_negative` n'est connecté à rien) : `prepare_workflows.py` crée le lien et `bind_workflow` ne le garde que si le plan fournit `exclude_points`, donc sans points rouges le graphe est celui déjà validé. **Les points rouges ne sont pas encore validés sur une vraie instance.** Le sujet doit être visible dès l'image 0.
- **`animate_move`** : seule la pose est reprise, le décor vient de l'image de référence (le masque SAM2 et le fond noirci sont retirés des trois étages `WanAnimateToVideo`, binding `background_inputs`). Pour changer de fond, générer le même personnage dans plusieurs décors (même `seed`, seul le lieu change dans le prompt Krea2) et rejouer la même vidéo pilote.
- **Pilote normalisé par l'app** (`app/driving_video.py`) : le gabarit recadre la vidéo en carré 640×640 (un clip 16:9 perd ses côtés) et lit les images 1:1 à la fréquence du plan (un clip 25 i/s lu à 16 i/s est au ralenti). L'app re-encode donc le pilote en carré à `fps` du plan ; `driving_fit: "pad"` met des bandes noires au lieu de recadrer. Les plans Animate sont carrés (`width == height`, 512 par défaut).
- **Sortie coupée à la longueur du pilote** : le dernier `SaveVideo` du gabarit est la version prolongée (149 images) qui reste figée sur la dernière pose, avec une coupe franche vers l'image 80, une fois le pilote terminé.
- **Chaîner deux passes** : `driving_video: {"job_id": "..."}` rejoue la sortie d'un job terminé (combat chat contre chat : la passe A remplace le premier combattant, la passe B remplace le second dans la sortie de A).
- **Mains** : déjà détectées. Le nœud `DWPreprocessor` du corps tourne avec `detect_hand=enable` (modèle `dw-ll_ucoco_384`, 21 points par main), un second ne gère que le visage ; passer sa résolution à 1024 ne change rien. Ce qui limite les doigts est la taille des mains dans l'image pilote : cadrer serré sur le haut du corps.

- **Pose humaine ou animale (`pose_source`)** : `human` (défaut) est la chaîne validée DWPose corps + mains + visage. `animal` remplace le squelette par celui d'`AnimalPosePreprocessor` (AP10K, 17 points de quadrupède, de `comfyui_controlnet_aux`), retire les recadrages de visage humain et calcule le point SAM2 vert avec le détecteur YOLOX animal (classes COCO 14 à 23) au lieu de yolov10m, qui ne trouve que des personnes. **Expérimental et non confirmé par la documentation** : ce nœud est un préprocesseur du ControlNet SD1.5 `animal-openpose`, la chaîne de prétraitement officielle de Wan 2.2 Animate est humaine (YOLOv10 + ViTPose + SAM2) et aucune source trouvée le 2026-10-03 ne dit que le modèle a été entraîné sur le squelette AP10K. Sur un chat qui suivait une vidéo de chat, le personnage de référence a repris les couleurs et suivi le mouvement, sans la tenue ni la posture debout. Un personnage assis ou debout comme une personne (chat anthropomorphe) se pilote mieux avec un acteur humain. Wan-Animate-2 (août 2026, nœud natif `WanAnimate2ToVideo`, sans squelette) accepte des personnages animaux mais seulement piloté par des humains dans ses exemples ; il n'est pas intégré (il demande un ComfyUI plus récent que celui figé ici). Pour un animal comme source de mouvement sans squelette humain, les contrôles profondeur/contours (Wan 2.2 Fun Control, VACE) ne dépendent pas de la catégorie.

`scripts/trend_cat_kungfu.py` automatise les recettes via l'API (jamais ComfyUI directement) : `transform` (un chat « réel » assis sur un canapé, ses pattes deviennent des mains par morph FLF2V entre deux images Krea2 de même graine, puis kung-fu en Move), `background` (même chat dans plusieurs décors côte à côte), `fight` (combat à deux chats en deux passes) et `sfx-download`/`sfx-mix` (cris et impacts Mixkit placés sur les pics de mouvement). Voir l'en-tête du script ; ses tests tournent sans GPU avec une API simulée. Exemples de payload : [examples/sequence-animate-move.json](examples/sequence-animate-move.json) et [examples/sequence-animate-mix-fight.json](examples/sequence-animate-mix-fight.json).

### Tendance « philosophe » (`scripts/trend_philosopher.py`) : route de production de base

Personnage sur un fond grandiose, voix, sous-titres. **Les règles validées le 2026-10-03 sont les valeurs par défaut** : il suffit de fournir le script, la voix, le personnage, les images de la rafale et la musique, sans aucune option de réglage.

```bash
python scripts/trend_philosopher.py --script script.txt --out final.mp4 --work-dir work \
    --character-image philosophe.png --no-matte --backgrounds fonds/*.png \
    --voice-wav voix.wav --voice-windows voix.json --music morceau.mp3
```

Ce que cette commande applique toute seule :

- **Un seul fond grandiose fixe** pendant presque toute la vidéo : `--grand-background image.png`, sinon il est **généré par Krea2** (`DEFAULT_GRAND_BACKGROUND_PROMPT`, graine 7, 832×1472, mis en cache dans `work/grand_background.png`), sinon, si l'API est injoignable, la première image de `--backgrounds`. Plus de changement permanent de fond (`--permanent-cuts` rétablit l'ancien comportement).
- **Rafale d'images juste avant le noir** : les autres fonds défilent, **0,2 à 0,4 s chacun** (environ 5 images à 24 i/s), 8 s par défaut, 10 s au maximum (`--flash-seconds`), calés sur les temps de la musique ; la rafale se termine là où commence l'écran noir.
- **Écran noir visé à 8 s** (`--black-screen-seconds`) vers 30-40 s, qui démarre sur une mesure de la musique et **se termine sur la fin de phrase la plus proche de début + 8 s** (`--no-black-end-on-sentence` pour une durée exacte) : l'image revient quand une phrase finit. Voix, sous-titres et musique continuent dessus ; les phrases qui le recouvrent restent affichées au centre. `--black-screen-at` le fixe, une valeur ≤ 0 le supprime. Un `blackdetect` le retrouve (avant 2026-10-03 le noir était masqué par le personnage).
- **Après le noir**, le fond grandiose revient et **le PNG du philosophe a disparu** (`--keep-character-after-black` pour le garder).
- **Flou léger sur les transitions seulement**, jamais sur les fonds eux-mêmes : une impulsion à 50 % qui retombe en 0,35 s au début de la rafale, puis une montée **courte (0,35 s) et progressive** (une marche par image, courbe lissée, jusqu'à 70 %) juste avant le noir et une descente symétrique juste après. Les sous-titres ne sont jamais floutés. `--blur 0` le coupe. Réalisé par des étapes `blend` à opacité constante et fenêtre `enable` (une expression par pixel faisait durer le rendu plus de 10 minutes ; il dure 65 s sur 12 cœurs).
- **Sous-titres style Hormozi** : majuscules épaisses (Montserrat ExtraBold, téléchargée par le Dockerfile, épinglée par commit et sha256 ; `TREND_CAPTION_FONT` pour changer), contour noir, 1 à 3 mots avec un petit effet de pop, taille selon l'impact, mot clé coloré (jaune le plus souvent, parfois rouge ou bleu, `CAPTION_KEY_COLORS`).
- **Sous-titres calés sur la voix** : les temps de chaque mot viennent de faster-whisper « small » (mots horodatés, apparié au texte du script, accents/casse/ponctuation ignorés, mots manqués interpolés, phrase mal reconnue = estimation) **dès que `faster-whisper` est installé** (`--no-align-words` pour s'en passer). Sur la voix du test (320 mots), 35 % des mots bougent de plus de 0,15 s par rapport à l'estimation proportionnelle, jusqu'à 0,69 s. Les sous-titres apparaissent 0,04 s avant le mot (`CAPTION_LEAD`).
- **Coupe en fin de phrase après 1 minute** (`--cut-after 60`, `0` pour garder toute la voix) : la vidéo s'arrête à la fin de la première phrase qui se termine après 60 s (+ 0,25 s de souffle) ; voix, sous-titres, musique (fondu de sortie) et plan des fonds sont calculés pour cette durée.
- **Musique** (`--music`, `--music-start`, `--music-gain-db`) : sous la voix avec sidechain, fondus, bouclée si trop courte ; `analyze_beats` (numpy seul) donne le tempo et la grille. Les droits d'un morceau du commerce sont à la charge de l'utilisateur.

**Réglages (valeur par défaut → comment la changer).** Chaque réglage existe sous trois formes : l'option de la ligne de commande, l'argument de `build_philosopher_trend(...)` (un seul appel de fonction), et, pour les constantes, la valeur en tête de `scripts/trend_philosopher.py`.

| Ce que ça règle | Défaut | Option CLI | Argument de la fonction |
|---|---|---|---|
| Durée visée de l'écran noir | 8 s | `--black-screen-seconds` | `black_screen_seconds` |
| Fin de l'écran noir sur une fin de phrase | oui (la plus proche de début + durée visée) | `--black-end-on-sentence` / `--no-black-end-on-sentence` | `black_end_on_sentence` |
| Tremblement après l'écran noir | aucun | `--after-black-shake rock\|drift\|...`, `--after-black-shake-amount` | `after_black_shake`, `after_black_shake_amount` (`permanent_shake` s'arrête alors à la fin du noir) |
| Échange de fonds grandioses après le noir | aucun | `--exchange-backgrounds a.png b.png c.png` | `exchange_backgrounds` : le fond grandiose revient avec l'image, puis ces fonds se succèdent (en boucle) |
| Sens des volets d'échange | `right up down left right` | `--exchange-directions` | `exchange_directions` : le nouveau fond se déplace dans le sens indiqué |
| Durée et courbe d'un volet d'échange | 1 s, `exponential` | `--exchange-seconds`, `--exchange-easing`, `--exchange-easing-strength` | `exchange_seconds`, `exchange_easing`, `exchange_easing_strength` |
| Instant de l'écran noir | auto (30-40 s, sur une mesure) | `--black-screen-at` (≤ 0 = pas de noir) | `black_screen_at` |
| Durée de la rafale d'images avant le noir | 8 s (max 10) | `--flash-seconds` | `flash_seconds` |
| Durée de chaque image de la rafale | 0,2 à 0,4 s | (constante `BG_FLASH_CUT_RANGE`) | |
| Style du flou autour du noir | `softfocus` (halo + fondu) | `--blur-style mix` pour l'autre | `blur_style` |
| Intensité du flou | 1.0 (0 = aucun) | `--blur` | `blur` |
| Transition entre les images de la rafale | `cut` | `--burst-transition cut\|shake\|wipe_left\|wipe_right\|swing` | `burst_transition` |
| Durée d'un volet | 0,10 s | `--burst-transition-seconds` | `burst_transition_seconds` |
| Force du shaking | 1,2 % de la largeur (zoom 1,05) | (constantes `SHAKE_AMPLITUDE`, `SHAKE_ZOOM`) | |
| Temps de chaque image de la rafale (jump cut) | automatique, 0,2 à 0,4 s sur les temps de la musique | `--flash-ratios 1,1,2,1` | `flash_ratios` (liste, une part de temps par image ; la rafale dure `flash_seconds` en tout) |
| Tremblement constant pendant la rafale | aucun | `--constant-shake jitter\|roll\|handheld\|sway\|rock` | `constant_shake` |
| Force du tremblement constant | 1.0 (le preset) | `--constant-shake-amount` | `constant_shake_amount` |
| Transition de l'écran noir (le fond ne change jamais) | `blur` (soft focus) | `--black-transition blur\|wipe_left\|wipe_right\|swing` | `black_transition` |
| Durée d'un volet | 0,8 s | `--black-wipe-seconds` | `black_wipe_seconds` |
| Courbe de vitesse du volet | `exponential` (lent puis accélère) | `--black-wipe-easing linear\|smooth\|exponential\|logarithmic`, `--black-wipe-easing-strength` | `black_wipe_easing`, `black_wipe_easing_strength` (défaut 4 en exponentiel, 9 en logarithmique) |
| Autre image de l'autre côté du volet | la première image de la rafale | `--black-wipe-panel-image` | `black_wipe_panel_image` |
| Luminosité de cette image (0 = noir pur) | 0,5, puis elle s'éteint 0,3 s avant le noir | `--black-wipe-panel-visibility` | `black_wipe_panel_visibility` (+ constante `BLACK_WIPE_PANEL_FADE_SECONDS`) |
| Tremblement pendant la transition du noir | aucun | `--transition-shake jitter\|roll\|handheld\|sway\|rock`, `--transition-shake-amount`, `--transition-shake-seconds` (1,2 s) | `transition_shake`, `transition_shake_amount`, `transition_shake_seconds` |
| Tremblement permanent, toute la vidéo | aucun | `--permanent-shake drift\|rock\|jitter\|roll\|handheld\|sway`, `--permanent-shake-amount` | `permanent_shake`, `permanent_shake_amount` (remplace `transition_shake`) ; tous les tremblements ne bougent que le fond, jamais le philosophe |
| Mode cinéma (bandes noires haut et bas) | désactivé | `--cinema` (moments par défaut), `--cinema-windows 23.8-38.7 55-65` | `cinema`, `cinema_windows` (liste de `(début, fin)` en secondes) |
| Début de vidéo en paysage | désactivé | `--landscape-start [SECONDES]` (4 s sans valeur) | `landscape_start_seconds` : bandes en place dès la 1re image, intro dessinée dans la bande, bandes parties à cette seconde |
| Largeur du philosophe (px, cadre de 1080) | 540 (c'était 760) | `--character-width` | `character_width` |
| Police de TOUS les sous-titres | Montserrat ExtraBold | `--caption-font constanb.ttf` (fichier ou nom de fichier) | `caption_font` (aussi la police du titre de fin par défaut) |
| Image de fin (la balance gothique / rock / romaine) | aucune | `--end-image scripts/assets/balance.png`, `--end-image-position center\|top\|bottom` (centre par défaut), `--end-image-height`, `--end-image-start`, `--end-image-fade` | `end_image`, `end_image_position`, `end_image_height`, `end_image_start`, `end_image_fade` |
| Titre de fin sur les bandes du cinéma | aucun | `--end-title "Ligne 1\|Ligne 2"`, `--end-title-font constanb.ttf` (obligatoire avec le titre) | `end_title`, `end_title_font` |
| Durée, effet et place du titre | 5 s, `shear_wave`, bande du haut | `--end-title-seconds`, `--end-title-effect shear_wave\|plain`, `--end-title-position top\|bottom\|center` | `end_title_seconds`, `end_title_effect`, `end_title_position` |
| Entrée des bandes | `slide` (elles glissent) ; `instant` = en place pendant tout le moment | `--cinema-mode slide\|instant` | `cinema_mode` |
| Courbe de vitesse des bandes | `exponential` (lent puis rapide) ; `logarithmic` = rapide puis freine | `--cinema-easing linear\|smooth\|exponential\|logarithmic`, `--cinema-easing-strength` | `cinema_easing`, `cinema_easing_strength` |
| Durée de l'entrée / sortie des bandes | 1,2 s | `--cinema-seconds` | `cinema_seconds` |
| Format de la bande d'image restante | 16:9 (écran de PC) | `--cinema-aspect 16:9\|2.39\|1.85` | `cinema_aspect` |
| Ouverture de la vidéo | `eyelid` (paupière) | `--intro-style eyelid\|oval\|none` | `intro_style` |
| Durée de l'ouverture | 1,2 s (+ 0,15 s de noir avant) | `--intro-seconds` | `intro_seconds` (+ constante `INTRO_HOLD_SECONDS`) |
| Vitesse d'arrivée du personnage | 2,0 s (plus petit = plus rapide) | `--character-rise-seconds` | `character_rise_seconds` |
| Délai avant son arrivée | 0 s | `--character-rise-delay` | `character_rise_delay` |
| Bord d'où il arrive | `bottom` | `--character-from bottom\|top` | `character_rise_from` |
| Où il s'arrête (0 haut, 0,5 milieu, 1 bas) | 0,55 (c'était 0,62 : un peu plus vers le centre) | `--character-final-y` | `character_final_y` |
| Disparition du personnage après le noir | oui | `--keep-character-after-black` pour le garder | `keep_character_after_black` |
| Coupe en fin de phrase après N s | 60 s | `--cut-after` (0 = pas de coupe) | `cut_after` |
| Alignement des mots par modèle | auto (si faster-whisper est installé) | `--align-words` / `--no-align-words` | `align_words` |
| Volume de la musique | -14 dB | `--music-gain-db` | `music_gain_db` |

Le jump cut seul, en une fonction : `build_jumpcut(images, out, total_seconds=5, ratios=[1, 1, 2, 1], width=1080, height=1920, transition="swing", constant_shake="handheld")` montre chaque image `ratios[i] / sum(ratios)` du temps total (1 s, 1 s, 2 s, 1 s ici). `jumpcut_lengths(total, ratios=...)` donne seulement les durées. Tremblements constants : `jitter` = rapide et petit, sans inclinaison ; `roll` = désaxage (l'image penche) ; `handheld` = caméra à l'épaule ; `sway` = balancement lent + tremblement rapide ; `rock` = balancement lent seul (1 Hz, ±0,5°), sans tremblement rapide et sans agrandir l'image (les bords découverts sont remplis par un miroir de la bordure), et sans micro-secousses : l'image est déplacée avec une interpolation au sous-pixel (filtre `perspective`) au lieu d'un recadrage par pixels entiers, qui laissait environ 0,5 px de crans ; mesuré sur une barre suivie image par image : 0,05 px d'écart avec une trajectoire lisse, contre 0,50 px avant. Ils s'ajoutent à n'importe quelle transition et durent toute la rafale (fondu d'entrée et de sortie de 0,12 s). `drift` est aussi lent et lisse que `rock` (rien au-dessus de 1 Hz, même amplitude, mêmes bords en miroir, même trajectoire au sous-pixel), mais sa direction ne se fixe jamais : trois ondes lentes sans rapport entre elles par axe (des rythmes différents en x et en y) et trois pour l'inclinaison (jusqu'à ±0,8°), si bien que l'image erre et que l'angle change sans cesse au lieu d'osciller sur une ligne. `permanent_shake="drift"` l'applique à toute la vidéo, de la première à la dernière image (l'image et le philosophe bougent ensemble, les sous-titres restent fixes, et le soft focus, l'ouverture en paupière et l'écran noir sont inchangés).

Mode cinéma : la vidéo verticale est mise en « letterbox » comme un écran horizontal, **à certains moments seulement** : une bande noire en haut et une en bas laissent une bande d'image au format `cinema_aspect` (16:9 = un écran de PC, soit 656 px de bande noire de chaque côté sur 1080×1920 ; 2.39 pour du scope). En `slide`, les bandes entrent par le haut et par le bas avec la courbe choisie (`exponential` : elles s'approchent lentement puis se referment vite ; `logarithmic` : elles arrivent vite puis freinent) et ressortent avec la même courbe à la fin du moment ; en `instant`, elles sont simplement en place pendant tout le moment. Avec `--cinema` sans fenêtres, les moments par défaut (`plan_cinema_windows`) sont la rafale d'images, du premier flash jusqu'à la fin de l'écran noir (les bandes ressortent sous le noir : l'image revient en portrait), et les 10 dernières secondes (les bandes entrent et restent jusqu'à la fin) : jamais la vidéo entière. Les bandes sont dessinées par-dessus l'image et le philosophe, sous l'écran noir et sous les sous-titres (qui restent donc lisibles, y compris sur les bandes). Le mode est désactivé par défaut tant qu'il n'est pas validé.

Début de vidéo en paysage : l'ouverture en ovale appartient à une scène paysage, pas à un plan portrait. Avec `landscape_start_seconds=4.0` (`--landscape-start`), la vidéo commence donc letterboxée : un moment du mode cinéma qui commence à 0 n'a pas d'entrée (les bandes sont en place dès la première image), l'intro (`oval` ou `eyelid`) est dessinée pour la bande visible (l'ovale est une ellipse large dans la bande 16:9, et non une ellipse haute sur tout le cadre portrait ; `intro_graph(..., band_height=...)`), puis les bandes partent (elles glissent avec la courbe choisie, ou d'un coup en mode `instant`) pour que l'image soit en portrait à la seconde indiquée. Le philosophe attend que les bandes commencent à partir avant d'entrer (`character_rise_delay` explicite et plus tardif : c'est lui qui gagne). Seul ce premier moment est ajouté ; `cinema=True` y joint la rafale et la fin. Un ovale demandé sans début en paysage fonctionne toujours mais affiche un avertissement.

La taille du philosophe se règle par `character_width` (540 px par défaut, 760 auparavant : 69 % de la hauteur du cadre pour 49 % maintenant).

Fin du noir et échanges de fonds : l'écran noir garde son début (sur une mesure de la musique) mais se termine sur la **fin de phrase la plus proche de début + 8 s** (`snap_black_end_to_sentence` : jamais une fin trop proche du début, ni à moins de 5 s de la fin de la vidéo) ; sur la voix de test, début 31,70 s → fin 40,72 s, soit 9,0 s, la fin de phrase la plus proche de 39,70 s. Après le noir, les fonds grandioses s'échangent par des volets qui glissent : `exchange_directions` (par défaut droite, haut, bas, gauche, droite) donne le sens dans lequel le nouveau fond **se déplace** (`right` : il entre par le bord gauche, `up` : par le bord bas, `down` : par le bord haut, `left` : par le bord droit), avec une courbe de vitesse (`exponential` par défaut, 1 s), l'ancien fond restant immobile dessous. Le premier fond est le fond grandiose (il revient avec l'image), puis `exchange_backgrounds` dans l'ordre et en boucle, jamais la même image deux fois de suite. Chaque volet **arrive** sur une fin de phrase quand il y en a une à moins d'une seconde de son moment régulier (`plan_exchange_times`), sinon il garde son rythme régulier ; deux volets ne sont jamais à moins de 1,5 s. `after_black_shake="rock"` donne ce tremblement lent à toute la phase d'après le noir (fond et volets bougent ensemble), `permanent_shake` s'arrêtant à la fin du noir.

Police des sous-titres et image de fin : `caption_font` (un fichier de police, ou un nom de fichier cherché dans les dossiers usuels) change la police de **tous** les sous-titres : le fichier est copié dans `work/fonts/` et donné à libass (`fontsdir`), et le nom de famille de la police est écrit dans le style ASS (`build_captions(font=...)`) ; sans lui, les sous-titres restent en Montserrat ExtraBold. C'est aussi la police du titre de fin si `end_title_font` n'est pas donné. `end_image` pose un PNG transparent près de la fin, en fondu, dès que les dernières bandes du cinéma sont en place. Par défaut il est **au centre de l'écran**, comme le philosophe, sur la bande d'image (92 % de sa hauteur) ; `end_image_position="top"` ou `"bottom"` le range dans une bande (70 % de sa hauteur). Les sous-titres s'écartent des lignes qu'il occupe : avec l'image au centre ils passent sur les bandes noires. `scripts/assets/balance.png` est un emblème **gothique, rock et romain** dessiné par `scripts/make_balance_png.py` (Pillow, aucun fichier externe ni licence) : une arche gothique à lancette avec crochets et rosace à vitrail rouge, une colonne ionique cannelée (chapiteau à volutes) qui porte le fléau aux pointes en fleur de lys, des chaînes à maillons, des plateaux à pointes, des lauriers romains au pied de la colonne ; sur les plateaux, un œil rayonnant (le regard, « être vu ») face à un cœur en flammes enlacé d'épines (une vie). Trait épais façon tatouage, os et noir, accents cramoisis, avec un halo sombre pour rester lisible sur n'importe quelle image.

Titre de fin (`scripts/title_card.py`) : les bandes noires des dernières secondes portent un titre. L'effet par défaut, `shear_wave` (d'après la miniature de référence « KNOWLEDGE AND WISDOM »), écrit le texte en **capitales grasses très espacées dont l'inclinaison ondule le long de la ligne** : une extrémité penche vers l'avant, le milieu reste droit, l'autre extrémité penche vers l'arrière, et la ligne suivante est le miroir de la précédente ; l'ondulation avance lentement (une période en 4 s) et le titre apparaît en fondu (0,6 s). `plain` donne le même texte droit. Les images sont écrites en PNG transparents (Pillow et numpy seulement) puis posées par `overlay` après les bandes et avant les sous-titres, qui s'écartent de la zone du titre (`build_captions(..., avoid=...)`). La police est un fichier (`--end-title-font`) ou un nom de fichier cherché dans `TREND_CAPTION_FONTSDIR` et les dossiers de polices usuels. Attention : Constantia est une police Windows (non redistribuable), absente de l'image Docker du serveur ; pour un rendu sur le serveur, il faut une police libre.

Le tremblement ne bouge que le fond : il est appliqué à la vidéo de fond avant que le philosophe y soit posé (`build_character_overlay(..., background_filters=...)`), donc le philosophe garde sa place dans le cadre pendant tout `permanent_shake` / `transition_shake`, et les sous-titres, ajoutés en dernier, aussi.

Volets de l'écran noir : un « rideau » plein cadre glisse sur l'image (position = expression du temps, donc n'importe quelle courbe ; seul `overlay` est utilisé, disponible en FFmpeg 4.4). `exponential` démarre très lentement puis accélère et arrive à pleine vitesse (le suspense), `logarithmic` part vite puis freine, `smooth` est lent-rapide-lent. Avec une image sur le rideau, le volet se termine 0,3 s avant le noir et l'image s'éteint vers le noir dans cet intervalle ; au retour, elle remonte du noir avant que le volet ne s'ouvre.

Volet qui ne passe qu'une fois (`black_wipes_graph(..., once=True)`) : le rideau glisse une seule fois et reste à l'écran, sans noir ni retour de la première image ; `panel_filters=` applique à l'autre image le même filtre que la première (par exemple la chaîne de `shake_filter(..., preset="rock")`), si bien que le balancement continue sans rupture d'une image à l'autre. Dans l'ancien enchaînement fermeture → noir → ouverture, l'autre image s'éteignait pendant `BLACK_WIPE_PANEL_FADE_SECONDS` (0,3 s) juste avant l'écran noir : c'était voulu, c'est ce que tu voyais passer au noir.

Sens des volets : `wipe_left` = le bord de la nouvelle image avance vers la gauche, `wipe_right` vers la droite, `swing` alterne gauche / droite à chaque coupe (balancement). Un volet se termine sur le temps prévu de la coupe, donc la musique reste calée sur la nouvelle image. Ouverture : `eyelid` = une fente en forme d'amande qui s'élargit comme une paupière, `oval` = une ellipse qui grandit depuis le centre.

`work/plan.json` consigne tout ce qui a été décidé : fenêtre noire, phases des fonds, coupe, mode d'alignement, tempo. Autres entrées possibles : fonds Krea2 (`--background-prompt`, `--auto-backgrounds N`), script écrit par un modèle de raisonnement local (`--topic`, Ollama `qwen3.6:27b`), voix précalculée (`--voice-wav` + `--voice-windows`) ou clonée (Qwen3-TTS, `--voice-ref`).

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
- Le générateur n’ajoute ni musique ni mixage audio ; les fondus actuels sont vidéo uniquement. Les bruitages des tendances chat sont ajoutés après coup par `scripts/trend_cat_kungfu.py`.
- Le plan `jumpcut` est validé en local avec FFmpeg 8.0.1 (rendu réel et job complet par l'API, `tests/test_jumpcut_shot.py`) ; le FFmpeg 8.1 statique de l'image Docker et son rendu sur le serveur loué n'ont pas encore été exécutés.
- Les workflows officiels évoluent. Le convertisseur échoue volontairement si un binding critique n’est plus identifiable.
- SQLite cible un seul conteneur et un seul worker ; une future exploitation multi-réplica nécessitera PostgreSQL et une file distribuée.
- Les licences CivitAI sont propres à chaque créateur et peuvent changer. Le champ `license_note` est informatif, pas un avis juridique.

Voir également [COMPATIBILITE.md](COMPATIBILITE.md), [DEPLOY_CLORE.md](DEPLOY_CLORE.md), [VERSIONS.md](VERSIONS.md) et [examples/sequence.json](examples/sequence.json).
