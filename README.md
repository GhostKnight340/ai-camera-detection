# Vigie Poste

Prototypes navigateur pour la detection de presence a un poste de travail,
a partir de la camera d'un telephone. Aucune image ne quitte l'appareil.

## Pages

| Fichier | Detection | Dependances |
|---|---|---|
| `index.html` | MediaPipe EfficientDet-Lite0, classe `person` | modele ~4 Mo charge depuis un CDN |
| `soustraction.html` | Soustraction de fond (CV classique) | aucune |

## Logique commune

Les deux pages partagent la partie qui compte reellement :

- **Zone poste** deplacable et redimensionnable (coordonnees normalisees)
- **Anti-rebond de 2 s** : l'etat doit tenir avant de basculer
- **Compteurs** : temps dans l'etat courant, arret cumule, nombre d'arrets
- `soustraction.html` ajoute une **chronologie d'occupation** sur 3 minutes

En production, seul le detecteur change (YOLOX / RF-DETR sur un boitier
Jetson en peripherie) ; la logique de zone et d'etat reste identique.

## Utilisation

La camera exige `https://` ou `localhost`. En local :

    python -m http.server 8000

Puis ouvrir http://localhost:8000/

## Limites connues

- `soustraction.html` detecte le **mouvement**, pas les personnes : un chariot
  qui passe declenche aussi la zone. Suffisant pour une camera fixe sur un
  poste fixe, pas pour une allee de circulation.
- Prototype de demonstration : ni multi-camera, ni persistance, ni stabilite
  eprouvee sur une journee complete.
