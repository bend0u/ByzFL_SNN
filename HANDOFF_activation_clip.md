# Handoff — Activation clipping & honest-side gradient shaping for Byzantine-robust FL

Document de passation pour reprendre ce travail dans une autre conversation (ex. Claude Code sur le serveur SSH). Résume le contexte, l'idée de recherche, ce qui est implémenté, les résultats mesurés, les conclusions, et les prochaines étapes.

---

## 1. Contexte projet

- Repo : **ByzFL_SNN** — benchmark PyTorch de Federated Learning robuste aux Byzantins (fork/extension de ByzFL, avec support SNN).
- Setup expérimental standard (toutes les heatmaps ci-dessous) :
  - Modèle `cnn_mnist` (2 conv + 2 FC, ReLU, `log_softmax`), MNIST, `NLLLoss`, SGD, lr **0.15**, momentum **0.9**, weight_decay **1e-4**, batch 128, **500 steps**, **5 seeds** (seeds = `training_seed + i`, base 42).
  - `n = 10` clients honnêtes, `f ∈ {0,1,2,3,4,5}` byzantins (axe X des heatmaps).
  - Hétérogénéité `γ ∈ {1.0, 0.66, 0.33, 0.0}` (axe Y ; γ=1 ≈ iid, γ=0 = non-iid extrême, classes quasi disjointes par client).
  - Pré-agrégation **NNM → ARC** ; agrégateurs **GeometricMedian, CenteredClipping, TrMean, MultiKrum**.
  - Attaques : **Optimal ALIE (neg1), SignFlipping, Optimal IPM**.
  - Heatmaps : `best_test` = worst-case sur toutes les attaques (et max sur agrégateurs pour la version "best") ; ou par agrégateur.

## 2. L'idée de recherche

**Question centrale : peut-on voir la fonction d'activation (et plus généralement des interventions côté client honnête) comme un paramètre qui contrôle la robustesse byzantine, orthogonal à l'agrégateur ?**

- Toute la littérature d'agrégation robuste (Krum, GM, TrMean, NNM, ARC, bucketing) a une borne d'erreur croissante avec la **variance / dissimilarité des gradients honnêtes** (terme σ² / bounded gradient dissimilarity). Donc : si on réduit cette variance côté honnête, **toute** borne d'agrégateur se resserre gratuitement — c'est un axe orthogonal au choix d'agrégateur.
- Notre angle = **côté honnête, pas côté serveur** (contrairement à ARC/NNM qui sont des filtres serveur voyant tous les vecteurs). C'est une *caractéristique du modèle/client*.
- Insight unificateur : le sweep de surrogate gradients SNN (atan/tri/box) qu'on a déjà fait **est déjà cette idée** — concevoir la forme de la dérivée d'une unité saturante pour que les stats de gradient honnête soient bornées. Le clipping d'activation CNN transplante la même idée du spike vers le ReLU. Cadre du papier : *unifier surrogate-gradient design (SNN) et activation clipping (CNN) comme le même mécanisme de "honest-side variance shaping", orthogonal à l'agrégation*.
- Précédent le plus proche dans la littérature FL : **El-Mhamdi, Guerraoui, Rouault — "Distributed Momentum for Byzantine-resilient SGD" (ICLR 2021)** — purement côté client (momentum), réduit la variance honnête, suffit à rendre une agrégation naïve bien plus robuste. Même philosophie, mécanisme différent (optimizer-level vs activation-level).

### Papiers à lire (avec pourquoi)
- **Allouah et al. 2024, "The Vital Role of Gradient Clipping in Byzantine-Resilient Distributed Learning"** (arXiv:2405.14432) — c'est la référence de `ARC` dans le code. ARC = clip relatif rank-based côté serveur (clippe au norm du k-ème plus grand, k = floor(2f/n·(n−f))). À lire pour la rigueur théorique (breakdown point, bornes hétérogénéité) que devrait viser notre version côté client.
- **Allouah et al. 2023, "Fixing by Mixing" (NNM), AISTATS** — la pré-agrégation NNM utilisée partout dans nos filenames. Sur CIFAR ils ne testent QUE α=1 ("moderate") et α=10 ("low") hétérogénéité, n=17, T=2000. Leurs baselines s'effondrent à ~10% (chance) sous ALIE+hétérogénéité — c'est exactement le mode d'échec qu'on observe.
- **Karimireddy, He, Jaggi 2022, "Bucketing" (ICLR)** — cadre théorique (variance reduction avant agrégateur robuste). Explique pourquoi γ (hétérogénéité) casse l'agrégation naïve.
- **Baruch et al. 2019, "A Little Is Enough" (NeurIPS)** — l'attaque ALIE. Perturbations petites/coordonnées qui restent dans la variance honnête → conçue pour passer sous les défenses. Explique pourquoi ALIE/IPM cognent plus fort que SignFlipping.
- **Choi et al. 2018, "PACT: Parameterized Clipping Activation"** — seuil de clip d'activation **appris** par backprop (+ pénalité L2), origine quantization. Mécanisme candidat pour "apprendre le clip naturellement".
- **Brock et al. 2021, "NFNets" / Adaptive Gradient Clipping (AGC)** — clip par-paramètre relatif au norm du poids : scale-invariant, transfère entre architectures. Corrige le problème "seuil calibré SNN pas adapté au CNN".
- **Andrew et al. 2021, "DP Learning with Adaptive Clipping" (NeurIPS)** — norme de clip adaptée en ligne via estimateur de quantile. Précédent du clip par quantile (mais motivé DP, pas robustesse).

### Potentiel de publication
- Angle "activation function as a robustness-controlling parameter, orthogonal to the aggregator" = axe sous-exploré, tient au moins pour un **workshop** aujourd'hui.
- Pour une conf pleine : il faut **(a) CIFAR** (MNIST seul ne passe pas en 2025+) et **(b) un hook théorique** reliant la forme de la dérivée d'activation au terme de variance des bornes d'agrégateur (argument Lipschitz : activation à dérivée bornée → gradient borné → σ² réduit).
- Venues : workshop FL/trustworthy-ML (NeurIPS/ICML/ICLR) → **SaTML** → **AISTATS** (même communauté que NNM).

## 3. Ce qui est implémenté

### Code source
- **`byzfl/fed_framework/clipped_activations.py`** (custom `autograd.Function`, mirrors `surrogates.py`) :
  - `ClippedReLU_STE` — forward `clamp(x,0,C)` ; backward dérivée = 1 pour tout `x>0` (ne s'annule jamais au-dessus de C).
  - `ClippedReLU_Ramp` — forward `clamp(x,0,C)` ; backward = 1 sur `(0,C]`, décroît linéairement à 0 sur `[C, rC]`, puis 0 (ratio `r=2`).
  - `AdaptiveQuantileClip` — forward `clamp(x,0,C)` avec `C = quantile_τ(x)` détaché (par forward pass) ; backward `plain` (0 au-dessus de C) ou `ste`. **Le variant STE a été DROP** (voir résultats).
- **`byzfl/fed_framework/models.py`** — modèles résolus par `getattr(models, name)`, pas de registry. Ajoutés :
  - `cnn_mnist_clip_ste_1` / `_2` (C=1,2)
  - `cnn_mnist_clip_ramp_1` / `_2` (C=1,2, r=2)
  - `cnn_mnist_clip_qcoord_plain_080` / `_090` (τ=0.8, 0.9)
  - (baseline existante `cnn_mnist_clipping_1/2/4` = hardtanh `F.hardtanh(x,0,C)`, dérivée VRAIE)
- **`byzfl/fed_framework/client.py`** — clip de norme de gradient côté client, deux positions INDÉPENDANTES via le helper commun `_clip_to_windowed_quantile(vector, history, q, window)` :
  - `grad_clip_quantile` / `grad_clip_window` → clippe le vecteur **POST-momentum** (copie ; le buffer momentum n'est jamais rescalé). Ne peut PAS empêcher le momentum de diverger.
  - `raw_grad_clip_quantile` / `raw_grad_clip_window` → clippe le gradient **RAW avant** l'accumulateur momentum. Même position que le `gradient_clip_val` fixe → son équivalent adaptatif. Prévient la divergence/NaN.
  - Helper : enregistre le norm PRÉ-clip (le quantile reflète la vraie distribution), **skip les norms non-finies** (un seul NaN empoisonnait sinon tous les quantiles suivants — seuil NaN → toute comparaison False → clip silencieusement désactivé). Fenêtre glissante car les norms de gradient sont non-stationnaires (décroissent en fin d'entraînement).
  - `gradient_clip_val` (FIXE, pré-existant) : `clip_grad_norm_(model.parameters(), val)` sur les `.grad` bruts juste après `backward()` — même position que `raw_grad_clip_quantile` mais seuil absolu fixe.

### Configs & scripts (après le refactor du repo)
- Configs : `configs/activation_clip/*.json`, générées par **`scripts/utils/generate_activation_clip_configs.py`**.
- Runner générique : **`scripts/experiments/run_activation_clip_sweep.py`** — expose `run_sweep(config, nb_jobs, distribute_gpus)` + CLI. Écrit les heatmaps dans **`results/activation_clip_plots/<variant>/`** (sous `results/`, seul arbre persistant/PVC-monté sur RunAI).
- Launchers : `scripts/launchers/rcp/*.sh` (RunAI/Docker) et `scripts/launchers/ssh/*` (dclgpusrv+tmux). Ex Python : `scripts/launchers/ssh/run_rawqclip080.py`.
- Rapport LaTeX : **`scripts/latex_generation/compile_activation_clip_report.py`** → **`reports/activation_clip_report.pdf`** (skip auto les panneaux sans data via `path_exists`).
- ⚠️ Le refactor a cassé des chemins (racine repo = 3 niveaux au-dessus depuis `scripts/*/`, pas 2 ; `latex_plots/` → `reports/`) — corrigés dans les commits récents.

### Infra RunAI / persistance plots (piège important)
- `Dockerfile` fait `COPY . ` → le code est **baké dans l'image**. RunAI tire l'image du registry et ne monte QUE `--pvc dcl-scratch:/home/bendouro/results`. Donc **tout changement de code nécessite rebuild + push de l'image**, et **seul `results/` persiste** (pod éphémère).
- Pod `results/` = login node `/mnt/dcl/scratch` (le préfixe `results/` disparaît côté login).
- Caching : `run_benchmark` → `eliminate_experiments_done` skippe tout setting `(seed,f,γ,agg,attack)` dont le fichier marqueur `train_time_tr_seed_X_dd_seed_Y.txt` existe. Donc reprendre un run ne recalcule que le manquant. Un run 2-seeds est un préfixe strict d'un run 5-seeds (seeds 42,43 réutilisés).

## 4. Résultats mesurés (MNIST, best_test worst-case sauf indication)

### Activations
| variante | clean (f=0) | γ=0 (f=1,2,3) | verdict |
|---|---|---|---|
| No-clip (ReLU) | 0.98 | 0.24, 0.21, 0.21 | référence |
| Hardtanh C=1 | 0.97 | **0.82, 0.66**, 0.16 | meilleur en hétérogénéité dure |
| Hardtanh C=2 | 0.98 | 0.80, 0.29, 0.21 | |
| STE C=1 | **0.84** | 0.84... s'effondre | **MAUVAIS même à f=0** |
| STE C=2 | **0.88** | | **MAUVAIS même à f=0** |
| Ramp C=1 | 0.98 | 0.79, 0.16, 0.16 | ≈ hardtanh, légèrement en-dessous |
| Ramp C=2 | 0.98 | 0.76, 0.16, 0.16 | ≈ hardtanh, légèrement en-dessous |

### Clips de norme (côté client)
best_test, γ=0.33 :
| | f=1 | f=2 | f=3 | f=4 | f=5 | clean |
|---|---|---|---|---|---|---|
| qclip momentum τ=0.80 (5 seeds) | 0.82 | 0.78 | 0.58 | 0.41 | 0.32 | **0.99** |
| rawqclip raw τ=0.80 (2 seeds) | 0.82 | 0.79 | 0.68 | 0.45 | 0.42 | 0.99 |

Comparaison propre **GM-vs-GM** (le run fixe-21 n'a QUE GeometricMedian) :
| GM, γ=0.33 | f=1 | f=2 | f=3 | f=4 | f=5 |
|---|---|---|---|---|---|
| qclip momentum (adaptatif) | 0.73 | 0.52 | 0.46 | 0.36 | 0.22 |
| rawqclip raw (adaptatif) | 0.75 | 0.55 | 0.63 | 0.41 | 0.29 |
| **grad-clip 21 (raw, FIXE)** | **0.89** | **0.89** | **0.74** | **0.65** | **0.68** |
| GM, γ=0 | 0.16 | 0.16 | 0.17(rawq) / 0.16 | — | — |
| **grad-clip 21, γ=0** | **0.77** | **0.45** | **0.40** | 0.16 | 0.16 |

### CIFAR-10 (runs séparés, lr 0.1, 5000 steps, n=10)
- **CIFAR ReLU** : clean ~0.83 ; à γ=0 s'effondre à ~0.10-0.13 sous ALIE/IPM (chance), SF plus doux. Cohérent avec la littérature (baselines NNM à ~10% sous ALIE+hétérogénéité). NB : 5000 steps > 2000 de la littérature, donc PAS un artefact de sous-entraînement — c'est un vrai mur.
- **CIFAR Tanh** : plafond clean à **0.81-0.82 même à f=0** → plafond de vanishing gradient. Confirme que tanh est une mauvaise piste pour tâche dure.

## 5. Conclusions établies

1. **STE = mauvaise idée pour un hard clip.** Dégrade même la clean accuracy (0.84/0.88 vs 0.98). Cause : gradient "fantôme" — la dérivée STE n'est celle d'aucune loss (non-conservative). Une unité saturée reçoit "pente 1" alors que le forward est plat → l'optimizer la pousse toujours plus loin en saturation → les activations se collent au plafond → capacité représentationnelle détruite. Confirmé par C=1 pire que C=2 (clip plus serré = plus d'unités dans la zone pathologique). **Droppé** (y compris le variant qcoord-STE).
2. **Ramp ≈ Hardtanh, hardtanh gagne dans les cellules dures.** À C=1 et C=2, la dérivée VRAIE (hardtanh) bat le ramp en hétérogénéité dure. Raison : le zéro de la dérivée fait **double emploi** — il borne le gradient honnête (bénéfice robustesse : σ² réduit) EN PLUS d'être consistant avec le forward. Le ramp laisse passer du gradient au-dessus de C → desserre cette borne. Sur MNIST (tâche facile, capacité de rab), le bénéfice "gradient borné" domine le coût "unités mortes". **Principe : la dérivée doit revenir à zéro dans la zone saturée (sinon phantom drift STE), mais pas brutalement au point de clip (sinon unités mortes).** Prédit un flip possible sur CIFAR (tâche dure) → **CIFAR = expérience discriminante ramp vs hardtanh, pas encore faite.**
3. **Clip de norme adaptatif (quantile) = ÉCHEC, et ce n'est PAS un problème de position.** L'expérience de désambiguïsation (rawqclip garde τ=0.80, change seulement raw↔momentum) a tranché :
   - **raw ≈ momentum** (quasi identiques, différences dans le bruit 2 vs 5 seeds).
   - **fixe ≫ adaptatif à position égale** (grad-clip-21 raw écrase rawqclip raw : GM γ=0 → 0.77 vs 0.16 à f=1).
   - Donc la vraie dichotomie est **fixe/absolu vs adaptatif/relatif**, PAS raw/momentum.
   - Cause : un quantile auto-référentiel ne peut pas imposer de borne absolue (le seuil monte si les gradients montent), clippe ~20% des pas par construction quelle que soit l'échelle, et **fait doublon avec ARC** (déjà relatif rank-based). Le 21 fixe apporte une **ancre d'échelle externe** que ni ARC ni NNM n'ont.
4. **Insight papier le plus propre** : "le clip côté client aide À CONDITION d'être une borne absolue, pas une statistique relative — sinon redondant avec l'agrégateur robuste." Résultat négatif structurant.
5. Le clip de norme fixe (21) **améliore aussi la clean accuracy** (stabilisation) — argument gratuit : le mécanisme ne coûte rien en accuracy propre (contrairement à tanh/STE).
6. **Mur γ=0, f≥3** : ~0.16 pour tout le monde. Aucun mécanisme testé ne franchit ce régime — même NNM+ARC ne le sauve pas. Probablement inutile d'y chercher un gain.

## 6. Prochaines étapes (recommandées, par priorité)

1. **[PRIORITÉ] Clip fixe calibré CNN.** Le 21 vient du SNN (pas adapté CNN) et gagne quand même → un fixe calibré CNN devrait faire ≥. Mesurer offline la distribution des norms de gradient raw d'un `cnn_mnist` honnête sur ~quelques centaines de steps, figer le seuil au **80e percentile mesuré une fois pour toutes** (ancre absolue, PAS recalculée en ligne). Garde l'idée "le seuil vient des données" SANS le défaut auto-référentiel. **Le lancer sur les 4 agrégateurs** (le 21 n'a que GM) pour comparaison best_test honnête.
2. **NE PAS** sweeper τ (0.7/0.9/0.95) ni compléter rawqclip à 5 seeds — le principe adaptatif est réfuté, pas un problème de réglage.
3. **CIFAR pour ramp vs hardtanh** — l'expérience qui décide si la dérivée-vraie gagne toujours ou si le ramp reprend l'avantage quand la capacité manque.
4. (Optionnel) Test d'orthogonalité hardtanh C=1 + clip fixe — mais le clip de norme adaptatif étant faible, prioriser le fixe calibré (point 1).
5. Pour le papier : formaliser le hook théorique (Lipschitz activation → gradient borné → σ² dans la borne d'agrégateur) + comparer/stacker avec le momentum d'El-Mhamdi et al.

## 7. État git (au moment de ce handoff)

- Branche `main`. Derniers commits pertinents : `5b5769460b` (rawqclip au rapport), `1233cf25d0` (plots rawqclip), `a997f6e36c` (restore report generator), `6b0e5475a7` (raw-gradient clip + fixes refactor), `5bf8902400` (refactor structure repo).
- ~3 commits locaux non poussés au moment d'écrire (dont ce handoff) — **penser à `git push` puis `git pull` sur le serveur**.
- Bugs corrigés notables : suffixe `_qclip_` dans le check de resume (le write side ne le produisait pas → reprise recalculait tout) ; chemins launchers/generator cassés par le refactor ; persistance plots (`plots/` éphémère → `results/activation_clip_plots/`).

## 8. Commandes utiles

```bash
# Regénérer les configs (idempotent)
python scripts/utils/generate_activation_clip_configs.py

# Lancer un sweep (SSH, ex A10)
./venv/bin/python3 scripts/experiments/run_activation_clip_sweep.py \
    --config configs/activation_clip/<variant>.json --distribute_gpus --nb_jobs 40

# Runner Python dédié (rawqclip pilote 2 seeds)
./venv/bin/python3 scripts/launchers/ssh/run_rawqclip080.py

# (Re)générer les heatmaps depuis des résultats déjà calculés, sans réentraîner
python scripts/plotting/plot_activation_clip_results.py --workers 4

# Compiler le rapport LaTeX
python scripts/latex_generation/compile_activation_clip_report.py   # -> reports/activation_clip_report.pdf
```
