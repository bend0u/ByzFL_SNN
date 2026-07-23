# STEP 0 — Localisation des vraies sources de robustesse (lecture seule)

**Correction par rapport à la première passe** : j'avais conclu à tort que les
grilles complètes (alpha=1.2 SNN, lr=0.15 CNN ReLU) n'existaient pas en local,
en ne cherchant que dans `results/` (fichiers per-run bruts). Elles n'y sont
effectivement pas — mais les **heatmaps déjà calculées** (celles qui alimentent
`robust_mixed_report.pdf`, via `scratch/compile_mixed_report.py`) existent bel
et bien sous `plots/`, en PDF vectoriel. Le sweep complet a bien tourné (comme
tu le disais) ; c'est le rapatriement depuis Docker qui n'a ramené que les
heatmaps déjà rendues, pas tous les fichiers `test_accuracy_*.txt` bruts.

## Sources retenues (confirmées, grille complète)

| Modèle | Dossier plots | Couverture |
|---|---|---|
| SNN atan (alpha=1.2) | `plots/robust_new_atan_sweep/alpha_1.2/` | 4 agrégateurs × 3 attaques × 6 f × 4 γ — **complet** |
| CNN ReLU (lr=0.15) | `plots/robust_comparison_sweep/learning_rate_0.15/` | idem — **complet** |
| CNN Tanh (lr=0.15) | `plots/cnn_tanh_heatmaps/` | idem — **complet** |

Ces PDF sont produits par `evaluate_results.test_heatmap()` avec `annot=True` :
chaque valeur affichée sur la heatmap est un objet texte vectoriel réel (pas un
pixel), donc extractible exactement via `pdftotext -bbox` — aucune saisie
manuelle, aucune OCR, aucune approximation.

## Extraction

Script : `analysis/geometry_v2/extract_heatmap_pdfs.py`. Pour chacun des 3
modèles × 4 agrégateurs (CenteredClipping, GeometricMedian, MultiKrum, TrMean)
× 3 attaques (ALIE, SignFlipping, IPM), il lit le fichier
`test_{attaque}_..._{agrégateur}_..._tolerated_f_equal_real.pdf` correspondant,
extrait les 24 valeurs (6 f × 4 γ) via leur position exacte sur la page, et les
place dans la bonne cellule (la géométrie de la grille — quelle ligne = quel γ,
quelle colonne = quel f — est déterminée directement à partir de la logique de
`test_heatmap()` dans `evaluate_results.py`, pas devinée).

**Résultat : 864/864 valeurs extraites (3 modèles × 4 agrégateurs × 3 attaques ×
24 cellules), aucun problème.** → `robustness_table_extracted.csv`.

## Double vérification

1. Valeurs relues à la main sur un extrait du PDF brut (TrMean, ALIE, SNN
   alpha=1.2, γ=0.0) : `[0.97, 0.76, 0.13, 0.14, 0.13, 0.13]` pour f=0..5 —
   identique à ce que le script a extrait.
2. Comparaison avec le fichier `test_..._TrMean_....pdf` **sans** suffixe
   d'attaque (celui que le repo calcule lui-même comme pire-cas sur les 3
   attaques, via `test_heatmap(target_attack=None)`) : **0 écart** sur les 24
   cellules par rapport au minimum que je recalcule moi-même sur les 3 attaques
   extraites séparément. Le pipeline d'extraction est donc cohérent avec la
   convention déjà utilisée dans le repo.

**Note méthodologique importante** : ces heatmaps utilisent la convention
`test_heatmap(metric="best_step")` du repo — accuracy au step qui maximise
l'accuracy de validation — et non ma moyenne ad hoc des 3 derniers checkpoints
utilisée dans la V1 de l'analyse (`analysis/geometry/`). C'est la convention
déjà établie dans ce repo (utilisée pour toutes les heatmaps existantes), donc
je la garde telle quelle plutôt que de la remplacer par mon propre choix.

## Ce qui devient inutile

Les templates `robustness_template_snn_atan12.csv` /
`robustness_template_cnn_relu.csv` (générés à l'étape précédente pour un
remplissage manuel) ne sont plus nécessaires — laissés sur disque au cas où,
mais `robustness_table_extracted.csv` les remplace avec des valeurs réelles et
exactes plutôt que manuelles.

---

**En attente de ta validation avant STEP 1** : le fichier
`robustness_table_extracted.csv` (colonnes `model, gamma, f, aggregator,
attack, accuracy`, 864 lignes) te semble correct pour construire la table de
robustesse finale (avec `acc_worst`, `acc_f5_worst`, `acc_f1to5_mean_worst`,
`robustness_drop` par (modèle, γ, agrégateur)) ?
