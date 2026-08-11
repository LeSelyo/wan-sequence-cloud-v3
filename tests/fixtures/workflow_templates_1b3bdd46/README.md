# Fixtures workflow_templates épinglées

Ces trois fixtures proviennent du commit
`1b3bdd46c945d54d893a3b43692d5963608fb7d4` de
`Comfy-Org/workflow_templates` :

- `video_wan2_2_14B_t2v.json` — SHA-256 brut officiel :
  `c07feedfb87de638cb34bd517d916fc7afa7786be453e433a7bb49833a0c2801` ;
- `video_wan2_2_14B_i2v.json` — SHA-256 brut officiel :
  `455337c85e3fb0c7da9b2e3e6408f02f4be3615ec9b77c0f18c4262b931dd650` ;
- `video_wan2_2_14B_flf2v.json` — SHA-256 brut officiel :
  `9fb579e07caff9081c14a4c0e3b983e210aa7d976f83f1c2758d2ad6ed949fdf`.

Les fichiers locaux ont seulement été réencodés en JSON compact ASCII afin de
rester stables sur Windows. Les tests comparent donc leur représentation JSON
canonique, en plus de vérifier que le Dockerfile épingle le commit ci-dessus.
Aucun modèle n'est inclus dans ces fixtures.
