# Méthodologie — Baseline de géométrie du gradient (f=0) et corrélation avec la robustesse

Ce document décrit précisément (1) la configuration exacte qui a tourné pour produire
les baselines de géométrie, et (2) comment les valeurs de robustesse ont été
sélectionnées parmi les sweeps Docker (qui, eux, couvrent plusieurs attaques et
plusieurs agrégateurs) pour construire la corrélation.

Scripts concernés : `analysis/geometry/analyze_geometry.py` (partie géométrie),
`analysis/geometry/robustness_correlation.py` (partie robustesse + corrélation).
Les deux sont en lecture seule sur `results/` — rien n'est relancé, tout est déjà sur disque.

---

## 1. Configuration exacte des baselines de géométrie (f=0)

Trois runs, un par modèle, définis par `configs/geometry_baseline/{snn_atan12,cnn_relu,cnn_tanh}.json`.
Tous les trois partagent :

| Paramètre | Valeur |
|---|---|
| `f` (Byzantins) | `[0]` — un seul point, pas de sweep |
| `attack` | `[NoAttack]` — un seul point (attaque nulle, cohérent avec f=0) |
| `aggregator` | `[TrMean]` — un seul agrégateur |
| `pre_aggregators` | `[NNM, ARC]` |
| `gamma` (`gamma_similarity_niid`) | `[1.0, 0.66, 0.33, 0.0]` — 4 valeurs, seul axe réellement balayé |
| `nb_training_seeds` | 1 (seed=42) |
| `nb_steps` | 500 |
| `momentum` / `weight_decay` / `batch_size` | 0.9 / 0.0001 / 128 |
| `size_train_set` | 0.8 |
| `store_client_vectors` | `false` (aucun dump de vecteurs bruts) |
| `store_per_client_metrics` | `true` |
| `store_models` | `false` |

Donc chaque config ne fait tourner que **4 runs** (un par γ) — il n'y a ni sweep d'attaque,
ni sweep d'agrégateur, ni sweep de f dans ces configs. C'est voulu : l'objectif était
un point de référence propre (honnête, non attaqué) par modèle et par γ, pas un sweep de robustesse.

Différences spécifiques par modèle :

| | SNN (`snn_atan12.json`) | CNN ReLU (`cnn_relu.json`) | CNN Tanh (`cnn_tanh.json`) |
|---|---|---|---|
| `model.name` | `cnn_mnist_snn` | `cnn_mnist` | `cnn_mnist_tanh` |
| `learning_rate` | 0.10 | 0.15 | 0.15 |
| `loss` | `ce_rate_loss` | `NLLLoss` | `NLLLoss` |
| SNN-spécifique | `beta=0.95`, `threshold=1.0`, `learn_threshold=false`, surrogate `atan` avec `alpha=1.2`, `time_steps=10`, encodage `constant` | — | — |

→ 12 runs au total (3 modèles × 4 γ), chacun de 500 steps, 1 seed. C'est exactement ce
qui est sous `results/geometry_baseline/{snn_atan12,cnn_relu,cnn_tanh}/`.

**Comment les métriques de géométrie sont calculées** : à chaque step d'entraînement,
juste après que chaque client honnête calcule son gradient post-momentum
(`get_flat_gradients_with_momentum()`), et **avant** l'agrégation (TrMean+NNM+ARC),
le hook (`byzfl/utils/gradient_geometry.py`, branché dans `train.py`) calcule N, Q, S,
`cos_mean`, les métriques par couche, et les histogrammes de `A_j` — puis les écrit dans
`metrics_geometry_tr_seed_42_dd_seed_42.csv` (501 lignes = steps 0 à 500, une ligne par step).

**Comment la valeur résumé par (modèle, γ) a été choisie** : médiane de chaque métrique
sur steps ∈ [100, 400] — fenêtre choisie après inspection visuelle des trajectoires
(régime transitoire les 100 premiers steps, puis plateau bruité pour γ<1.0 ; pour γ=1.0,
S continue de décroître lentement sur toute la fenêtre — documenté comme limite dans le rapport).

---

## 2. Sélection des valeurs de robustesse (sweeps Docker importés)

Contrairement aux baselines de géométrie, les gros sweeps de robustesse (tournés sur
Docker, puis en partie importés dans ce repo) balaient **plusieurs agrégateurs**
(`GeometricMedian`, `CenteredClipping`, `TrMean`, `MultiKrum`) **et plusieurs attaques**
(`Optimal_ALittleIsEnough_neg1`, `SignFlipping`, `Optimal_InnerProductManipulation`),
en plus de f et γ. Pour construire un score de robustesse unique par (modèle, γ, f),
il fallait donc choisir *lequel* de ces agrégateurs/attaques utiliser, puisque tout n'a
pas été importé pour toutes les combinaisons.

### 2.1 Choix de l'agrégateur

J'ai vérifié, pour chaque agrégateur, combien de dossiers de résultats existent
réellement en local pour chaque f (0 à 5), croisé sur les 3 modèles. Résultat :

- `TrMean` : complet uniquement pour CNN Tanh (`tanh_heatmap_sweep`). Absent pour le SNN
  (le sweep atan importé ne contient que `CenteredClipping` et `GeometricMedian`).
  Pour CNN ReLU, seulement lr=0.1 disponible et incomplet au-delà de f=2.
- `GeometricMedian` : **complet pour les 3 modèles** (f=0..5, les 4 γ, toutes les
  attaques importées) — c'est le seul agrégateur commun avec couverture complète.

→ **`GeometricMedian`+`NNM`+`ARC` a été choisi pour les 3 modèles**, uniquement parce
que c'est le seul choix qui donne une grille complète partout. Ce n'est pas le même
agrégateur que celui de la baseline géométrie (`TrMean`) — c'est documenté comme
approximation dans le rapport (§ "Robustness sources").

### 2.2 Choix de l'attaque, et calcul du score par (modèle, γ, f)

Pour chaque cellule (modèle, γ, f), le score de robustesse est le **pire cas parmi les
attaques disponibles** (minimum de l'accuracy sur les attaques importées) — c'est la
même convention que la fonction `test_heatmap()` déjà présente dans le repo
(`byzfl/benchmark/evaluate_results.py`), qui définit la robustesse comme la performance
sous l'attaque la plus dommageable.

- **CNN ReLU et CNN Tanh** : les 3 attaques (ALIE, SignFlipping, IPM) sont disponibles
  localement → vrai pire-cas sur 3 attaques.
- **SNN** : seule `Optimal_ALittleIsEnough_neg1` a été importée pour la combinaison
  `GeometricMedian`+alpha=1.25 → le "pire cas" se réduit à cette unique attaque.
  Documenté explicitement comme limitation (pas un vrai pire-cas sur 3 attaques pour le SNN).

Pour chaque (attaque, f, γ), la valeur d'accuracy elle-même est : la moyenne de
`test_accuracy` sur les 3 derniers checkpoints d'évaluation (steps 400, 450, 500 —
choisi pour lisser le bruit sans juste prendre le pic max), moyennée sur les 5 seeds
d'entraînement disponibles dans ces sweeps Docker (`nb_training_seeds=5`, contre 1 seule
seed pour nos baselines f=0, car f=0 est un point stable qui n'a pas besoin de 5 seeds).

### 2.3 Choix du lr / alpha (SNN et CNN ReLU)

Le lr (CNN) et le alpha (SNN) de la baseline géométrie ne sont pas toujours disponibles
dans les données importées. J'ai choisi, pour chaque modèle, la valeur la plus proche
de la baseline **parmi celles qui ont une couverture complète f=0..5** :

| Modèle | Dossier utilisé | lr / alpha choisi | Écart vs baseline géométrie |
|---|---|---|---|
| SNN | `results/snn/robust_new_atan_sweep` | alpha=1.25 | baseline = 1.2 (1.2 n'a pas été importé, seuls {0.5,0.75,1.0,1.25,1.5,2.0,3.0} le sont) |
| CNN ReLU | `results/cnn/weekend` | lr=0.05 | baseline = 0.15 (l'import à lr=0.15 s'arrête à f=2) |
| CNN Tanh | `results/cnn/tanh_heatmap_sweep` | lr=0.15 | **exact**, aucun écart |

---

## 3. Résumé du flux de données

```
results/geometry_baseline/{modèle}/...        <- f=0, NoAttack, TrMean, 1 seed
        │  (métriques N,Q,S,cos_mean par step, calculées en ligne)
        ▼
analyze_geometry.py  →  summary_table.csv     <- médiane [100,400] par (modèle, γ)
        │
        ▼
robustness_correlation.py
        │  lit aussi : results/snn/robust_new_atan_sweep,
        │              results/cnn/weekend,
        │              results/cnn/tanh_heatmap_sweep
        │  (GeometricMedian+NNM+ARC, pire-cas sur attaques disponibles,
        │   f=0..5, moyenne 5 seeds, moyenne des 3 derniers checkpoints)
        ▼
geometry_robustness_merged.csv, correlation_table.csv, correlation_scatter.png
```

Rien n'a été relancé pour produire ce document ni les figures : tout vient de fichiers
déjà présents sur disque (baselines géométrie + sweeps Docker importés).
