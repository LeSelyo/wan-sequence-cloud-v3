# Déploiement permanent (IP/domaine fixes)

Ce document couvre le cas d'un serveur loué durablement (pas une location
éphémère Clore/RunPod/Verda dont l'IP change à chaque location). Le reste du
déploiement (variables d'environnement, modèles, LoRA) reste celui décrit dans
`DEPLOY_CLORE.md` ; seul l'accès réseau change ici.

## Pourquoi Caddy + TLS ici, et pas sur une location éphémère

- Sur Clore, le port "HTTP" transite par leur propre proxy (`clorecloud.net`) :
  ils déchiffrent puis retransmettent en clair vers le conteneur. Impossible d'y
  superposer un certificat Let's Encrypt à nous, puisque leur défi HTTP-01 exige
  un domaine qui pointe en permanence sur l'IP louée — laquelle change à chaque
  location.
- Sur un serveur permanent, ce problème disparaît : le domaine pointe une fois
  pour toutes sur l'IP fixe, et Caddy obtient et renouvelle son propre certificat.

## Ce que fait `deploy/Caddyfile`

- Termine le TLS **sur cette machine** : ni Clore, ni un autre tiers, ne voit
  jamais le token bearer ou le trafic ComfyUI en clair.
- Ne laisse que 443 (et 22 pour SSH) ouverts sur le pare-feu ; l'app (8000) et
  ComfyUI (8188) restent liés à `127.0.0.1`, jamais exposés directement.
- L'app FastAPI garde son authentification bearer existante (`API_TOKEN`),
  Caddy ne fait que relayer.
- **ComfyUI n'a aucune authentification à lui** — Caddy lui ajoute un mot de
  passe HTTP Basic (`basicauth`), seule protection avant l'interface complète
  (graphe de noeuds, édition manuelle des workflows).

## Mise en place

```bash
# Sur le serveur, une fois le domaine pointé sur son IP :
apt-get install -y caddy
caddy hash-password --plaintext 'un mot de passe fort'   # -> COMFYUI_BASIC_HASH

export API_HOST=wan.example.com
export COMFYUI_HOST=comfyui.wan.example.com
export COMFYUI_BASIC_USER=admin
export COMFYUI_BASIC_HASH='<sortie de caddy hash-password>'
caddy run --config deploy/Caddyfile
```

Le mot de passe ComfyUI ne doit jamais être tapé dans un champ de site tiers ni
committé — seul le hash (`COMFYUI_BASIC_HASH`) circule, jamais le mot de passe
en clair.

## Sur une location éphémère (Clore, en attendant un serveur permanent)

Pas de Caddy ici — le tunnel SSH est la méthode : il réutilise la clé SSH déjà
en place pour la location, n'ouvre aucun port public supplémentaire, et donne
l'interface ComfyUI complète en local, chiffrée de bout en bout :

```bash
ssh -i ~/.ssh/id_ed25519_clore -p <port_ssh> -L 8188:127.0.0.1:8188 root@<hote_clore>
# puis ouvrir http://127.0.0.1:8188 dans son propre navigateur
```
