# Rapport de compatibilité des éléments fournis

Vérification effectuée à partir des métadonnées CivitAI et des descriptions publiées le 26 juillet 2026.

| Élément | Type réel | Base déclarée | Décision |
|---|---|---|---|
| Round world chubby animals | LoRA | SD 1.5 | Conservée pour générer une image/keyframe SD 1.5 uniquement. Jamais dans Wan. |
| WAN 2.2 I2V GGUF — 8GB Daily Workflow | Workflow | Wan 2.2 I2V-A14B | Exclu du catalogue LoRA. C’est un workflow ComfyUI avec modèles GGUF et nœuds communautaires. |
| WAN2.2 Spatial Magic v2 | LoRA | Wan 2.2 I2V-A14B | Conservée. L’auteur indique explicitement `wan2.2_high`, déclencheur `kjmf magic`, poids conseillé 1.0. |
| AnimaIka | Checkpoint | Anima | Exclu du catalogue LoRA. C’est un modèle d’image complet, pas une LoRA vidéo. |
| TFM Game Girl 7 Pixel | Deux LoRA | Wan 2.2 T2V-A14B | Conservées sous deux IDs : high-noise et low-noise. Elles doivent être utilisées ensemble si l’on veut affecter les deux experts. |
| Illustrious Pixel Art | LoRA | Illustrious / Pony / SD 1.5 selon version | Versions séparées. Utilisation image/keyframe seulement avec la base exacte. |
| M_Pixel | LoRA | SD 1.5 | Conservée pour image/keyframe SD 1.5 seulement. |
| Damage Pixel Art | LoRA | SD 1.5 | Conservée pour image/keyframe, mais marquée non commerciale par prudence car le champ d’usage commercial CivitAI est vide. |
| AnimaliaStyle | LoRA | SD 1.5 | Conservée pour image/keyframe SD 1.5 seulement. L’animation se fait ensuite par Wan I2V. |
| Lin LoRA v3 HIGH | LoRA | Wan 2.2 T2V-A14B | Conservée sur l’expert high-noise uniquement. La fiche la décrit simplement comme « Character LoRA » : rien ne permet de l’identifier comme une LoRA générique selfie/caméra. |

## Réponse courte à « tout peut s’utiliser partout ? »

Non. Une LoRA contient des corrections destinées à des couches et dimensions précises du modèle sur lequel elle a été entraînée. Une LoRA SD 1.5 ne correspond pas aux couches de Wan 2.2. Même dans Wan 2.2, T2V et I2V ne sont pas interchangeables, et l’architecture MoE A14B possède des experts high-noise et low-noise distincts.

Le chemin sûr pour une LoRA d’image est :

```text
checkpoint image compatible + LoRA image → keyframe PNG → Wan 2.2 I2V → vidéo
```

Cela transfère visuellement le style par l’image initiale ; cela ne transforme pas la LoRA d’image en LoRA vidéo.

## Éléments non inférés

« Character sheet », flou de fond, portrait royal, affiche d’idole, selfie et accessoires ne sont pas des compatibilités techniques garanties par une LoRA. Ce sont des objectifs de cadrage et de contenu à exprimer dans le prompt, à tester avec des seeds fixes, puis à valider visuellement. Le catalogue n’ajoute donc aucune capacité qui ne figure pas dans la fiche du créateur.
