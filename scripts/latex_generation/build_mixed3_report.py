#!/usr/bin/env python3
"""
Build a per-aggregator / per-attack comparison report (robust_mixed style) for
three models: CNN ReLU grad-clip C=21, CNN ReLU self-clip w1, SNN Atan alpha=1.2.
Copies every needed heatmap into reports/mixed3_assets/ with clean names, then
emits reports/robust_mixed3_report.tex (self-contained, compiles after a git pull).
"""
import os, shutil

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSETS = os.path.join(REPO, "reports", "mixed3_assets")
os.makedirs(ASSETS, exist_ok=True)
SUF = "gamma_similarity_niid_NNM_ARC{agg}_nb_honest_clients_10_tolerated_f_equal_real.pdf"

# model_key -> (source_dir, model_token_in_filename, display_name)
MODELS = {
    "g21": (os.path.join(REPO, "activation_clip_plots/activation_clip_plots/cnn_mnist_gradclip21"),
            "cnn_mnist", r"CNN ReLU, Grad-Clip $C{=}21$"),
    "w1":  (os.path.join(REPO, "results/activation_clip_plots/cnn_mnist_selfclip_w1"),
            "cnn_mnist", r"CNN ReLU, Self-Clip \texttt{w1}"),
    "snn": (os.path.join(REPO, "plots/robust_new_atan_sweep/alpha_1.2"),
            "cnn_mnist_snn", r"SNN Atan ($\alpha{=}1.2$)"),
}
ATTACKS = {"merged": "", "ALIE": "Optimal_ALittleIsEnough_neg1",
           "SF": "SignFlipping", "IPM": "Optimal_InnerProductManipulation"}
ANAME = {"merged": "Worst-Case Across Attacks", "ALIE": "Optimal ALIE",
         "SF": "Sign Flipping", "IPM": "Optimal IPM"}
AGGS = {"best": None, "GeometricMedian": "GM", "CenteredClipping": "CC",
        "TrMean": "TrMean", "MultiKrum": "MultiKrum"}


def src_name(model_token, attack_tok, agg):
    atk = f"_{attack_tok}" if attack_tok else ""
    if agg == "best":
        return f"best_test{atk}_mnist_{model_token}_" + SUF.format(agg="")
    aggpart = f"_{agg}"
    return f"test{atk}_mnist_{model_token}_" + SUF.format(agg=aggpart)


def asset(mk, atk_key, agg):
    return f"{mk}_{agg}_{atk_key}.pdf"


def copy_all():
    missing = []
    for mk, (d, tok, _) in MODELS.items():
        for atk_key, atk_tok in ATTACKS.items():
            for agg in AGGS:
                s = os.path.join(d, src_name(tok, atk_tok, agg))
                dst = os.path.join(ASSETS, asset(mk, atk_key, agg))
                if os.path.isfile(s):
                    shutil.copy(s, dst)
                else:
                    missing.append(s)
    return missing


def fig_row(atk_key, agg, caption):
    """A figure: the 3 models side by side for one (attack, aggregator) view."""
    subs = []
    for mk in ("g21", "w1", "snn"):
        subs.append(
            r"    \begin{subfigure}[b]{0.32\textwidth}\centering"
            f"\n        \\includegraphics[width=\\textwidth]{{{asset(mk, atk_key, agg)}}}"
            f"\n        \\caption{{{MODELS[mk][2]}}}\n    \\end{{subfigure}}")
    body = "\n    \\hfill\n".join(subs)
    return (r"\begin{figure}[htbp]\centering" + "\n" + body +
            f"\n    \\caption{{{caption}}}\n\\end{{figure}}\n")


def build_tex():
    L = []
    L.append(r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=1.6cm]{geometry}
\usepackage{graphicx}\usepackage{subcaption}\usepackage{amsmath}
\usepackage[colorlinks=true,linkcolor=blue]{hyperref}
\graphicspath{{mixed3_assets/}}
\title{\textbf{Byzantine-Robust FL: CNN Grad-Clip 21 vs CNN Self-Clip w1 vs SNN Atan 1.2}\\
\large Per-aggregator and per-attack heatmap comparison}
\author{ByzFL\_SNN}\date{\today}
\begin{document}\maketitle

\noindent Three honest-side configurations on \texttt{cnn\_mnist}/\texttt{cnn\_mnist\_snn}:
(i) CNN ReLU with a fixed raw-gradient clip $C{=}21$, (ii) CNN ReLU with the self-calibrated
``clip to the first gradient'' rule (\texttt{w1}), (iii) SNN with the ArcTangent surrogate
($\alpha{=}1.2$). Shared grid: $n{=}10$ honest, $f\in\{0,\dots,5\}$, heterogeneity
$\gamma\in\{1,0.66,0.33,0\}$, pre-agg NNM$\to$ARC, 5 seeds. Test accuracy; rows $=\gamma$,
columns $=f$.
\clearpage
""")

    # Section 1: best over aggregators, worst over attacks + per attack
    L.append(r"\section{Best Over Aggregators}")
    L.append(fig_row("merged", "best",
                     "Best test accuracy, worst-case across attacks (best over aggregators)."))
    for atk in ("ALIE", "SF", "IPM"):
        L.append(fig_row(atk, "best",
                         f"Best test accuracy under {ANAME[atk]} (best over aggregators)."))
    L.append(r"\clearpage")

    # Section 2..: per aggregator, each with merged + 3 attacks
    for agg, short in AGGS.items():
        if agg == "best":
            continue
        L.append(f"\\section{{Aggregator: {agg}}}")
        L.append(fig_row("merged", agg,
                         f"{agg}: worst-case across attacks."))
        for atk in ("ALIE", "SF", "IPM"):
            L.append(fig_row(atk, agg, f"{agg} under {ANAME[atk]}."))
        L.append(r"\clearpage")

    L.append(r"\end{document}")
    return "\n".join(L)


def main():
    missing = copy_all()
    n = len([f for f in os.listdir(ASSETS) if f.endswith(".pdf")])
    print(f"copied {n} figures into {ASSETS}")
    if missing:
        print(f"WARNING: {len(missing)} missing sources:")
        for m in missing[:10]:
            print("  ", os.path.relpath(m, REPO))
    tex = build_tex()
    out = os.path.join(REPO, "reports", "robust_mixed3_report.tex")
    with open(out, "w") as f:
        f.write(tex)
    print("wrote", out)


if __name__ == "__main__":
    main()
