import os
import subprocess

# Configs with actual plot data right now (activation_clip_plots/ + existing baselines).
# Reference-hardtanh and tanh/no-clip baselines reuse plots already produced by the
# original MNIST clipping sweep; STE/Ramp variants reuse the new activation_clip_plots/.
# All configs mixed together in every grid (no more per-clip-value sectioning).
CONFIGS = [
    dict(key="noclip", caption="No Clip (ReLU)",
         dir="../plots/mnist_clipping_heatmaps/cnn_mnist", model="mnist_cnn_mnist"),
    dict(key="tanh", caption="Tanh",
         dir="../plots/cnn_tanh_heatmaps", model="mnist_cnn_mnist_tanh"),
    dict(key="gradnorm21", caption="Grad-Norm Clip $=21$",
         dir="../plots/cnn_clipped_heatmaps", model="mnist_cnn_mnist"),
    dict(key="hardtanh_1", caption="Hardtanh $C{=}1$",
         dir="../plots/mnist_clipping_heatmaps/cnn_mnist_clipping_1",
         model="mnist_cnn_mnist_clipping_1"),
    dict(key="hardtanh_2", caption="Hardtanh $C{=}2$",
         dir="../plots/mnist_clipping_heatmaps/cnn_mnist_clipping_2",
         model="mnist_cnn_mnist_clipping_2"),
    dict(key="ste_1", caption="STE $C{=}1$",
         dir="../activation_clip_plots/cnn_mnist_clip_ste_1",
         model="mnist_cnn_mnist_clip_ste_1"),
    dict(key="ste_2", caption="STE $C{=}2$",
         dir="../activation_clip_plots/cnn_mnist_clip_ste_2",
         model="mnist_cnn_mnist_clip_ste_2"),
    dict(key="ramp_1", caption="Ramp $C{=}1$ ($r{=}2$)",
         dir="../activation_clip_plots/cnn_mnist_clip_ramp_1",
         model="mnist_cnn_mnist_clip_ramp_1"),
    dict(key="ramp_2", caption="Ramp $C{=}2$ ($r{=}2$)",
         dir="../activation_clip_plots/cnn_mnist_clip_ramp_2",
         model="mnist_cnn_mnist_clip_ramp_2"),
    # Gradient-norm clip (NOT an activation): plain ReLU cnn_mnist, so its files
    # carry the same "mnist_cnn_mnist" token as No-Clip -- the folder disambiguates.
    dict(key="qclip_080", caption="Norm Clip $\\tau{=}0.80$ (momentum)",
         dir="../activation_clip_plots/cnn_mnist_qclip_080",
         model="mnist_cnn_mnist"),
]

SUFFIX = "gamma_similarity_niid_NNM_ARC_nb_honest_clients_10_tolerated_f_equal_real"

ATTACKS = [
    dict(key="alie", tag="Optimal_ALittleIsEnough_neg1", title="Optimal ALIE Attack", short="ALIE"),
    dict(key="sf", tag="SignFlipping", title="Sign Flipping Attack", short="SF"),
    dict(key="ipm", tag="Optimal_InnerProductManipulation", title="Optimal IPM Attack", short="IPM"),
]

AGGREGATORS = [
    dict(key="cc", tag="CenteredClipping", title="Centered Clipping (CC)", short="CC"),
    dict(key="gm", tag="GeometricMedian", title="Geometric Median (GM)", short="GM"),
    dict(key="mk", tag="MultiKrum", title="Multi-Krum (MK)", short="MK"),
    dict(key="tm", tag="TrMean", title="Trimmed Mean (TM)", short="TM"),
]


def best_test_path(cfg, attack_tag=None):
    parts = ["best_test"]
    if attack_tag:
        parts.append(attack_tag)
    parts.append(cfg["model"])
    parts.append(SUFFIX)
    return f"{cfg['dir']}/{'_'.join(parts)}.pdf"


def test_path(cfg, agg_tag, attack_tag=None):
    parts = ["test"]
    if attack_tag:
        parts.append(attack_tag)
    parts.append(cfg["model"])
    parts.append(SUFFIX.replace("gamma_similarity_niid_NNM_ARC",
                                 f"gamma_similarity_niid_NNM_ARC_{agg_tag}"))
    return f"{cfg['dir']}/{'_'.join(parts)}.pdf"


def path_exists(rel_path, latex_dir):
    return os.path.exists(os.path.join(latex_dir, rel_path))


def subfig(width, path, caption, label):
    return "\n".join([
        r"    \begin{subfigure}[b]{%s\textwidth}" % width,
        r"        \centering",
        f"        \\includegraphics[width=\\textwidth]{{{path}}}",
        f"        \\caption{{{caption}}}",
        f"        \\label{{fig:{label}}}",
        r"    \end{subfigure}",
    ])


def make_grid(paths_captions_labels, width, caption, label, per_row=3, latex_dir=None):
    available = [(p, c, l) for (p, c, l) in paths_captions_labels
                 if latex_dir is None or path_exists(p, latex_dir)]
    if not available:
        return f"% [no data available yet for {label}]\n"
    lines = [r"\begin{figure}[htbp]", r"    \centering"]
    for i, (path, cap, lab) in enumerate(available):
        lines.append(subfig(width, path, cap, lab))
        if i != len(available) - 1:
            if (i + 1) % per_row == 0:
                lines.append(r"    \vspace{0.3cm}")
            else:
                lines.append(r"    \hfill")
    lines.append(f"    \\caption{{{caption}}}")
    lines.append(f"    \\label{{fig:{label}}}")
    lines.append(r"\end{figure}")
    return "\n".join(lines)


def main():
    # This file lives at <repo>/scripts/latex_generation/, so the repo root is 3 levels up.
    workspace_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    latex_dir = os.path.join(workspace_dir, "reports")
    os.makedirs(latex_dir, exist_ok=True)
    tex_path = os.path.join(latex_dir, "activation_clip_report.tex")

    with open(tex_path, "w") as f:
        f.write(r"""\documentclass{article}
\usepackage{graphicx}
\usepackage{geometry}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{subcaption}
\usepackage{amsmath}
\usepackage{amssymb}

\geometry{a4paper, margin=0.65in}

\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    filecolor=magenta,
    urlcolor=cyan,
}

\title{Gradient-Preserving Activation Clipping: MNIST Report}
\author{ByzFL SNN Benchmark}
\date{\today}

\begin{document}

\maketitle

\clearpage
\tableofcontents
\clearpage

\section{Activation Definitions}

All variants below share the same MNIST CNN architecture
(\texttt{Conv2d(1,20,5,1)} $\to$ \texttt{Conv2d(20,50,5,1)} $\to$ \texttt{Linear(800,500)}
$\to$ \texttt{Linear(500,10)}), differing only in the activation applied after the first
three layers (the final layer always uses \texttt{log\_softmax}). Forward/backward
definitions below are transcribed directly from
\texttt{byzfl/fed\_framework/clipped\_activations.py} and
\texttt{byzfl/fed\_framework/client.py}.

\subsection{Baseline: Hardtanh Clip (existing, true derivative)}
Forward and backward are both the standard \texttt{hardtanh} clamp -- the gradient
genuinely vanishes above the clip value:
\begin{align*}
    y &= \mathrm{clamp}(x, 0, C) \\
    \frac{\partial y}{\partial x} &=
    \begin{cases}
        1 & 0 < x < C \\
        0 & \text{otherwise}
    \end{cases}
\end{align*}

\subsection{STE Clip (\texttt{ClippedReLU\_STE})}
Forward is the same hard clamp; the backward pass is a straight-through estimator that
never zeroes the gradient for positive inputs, regardless of saturation:
\begin{align*}
    y &= \mathrm{clamp}(x, 0, C) \\
    \frac{\partial y}{\partial x} &=
    \begin{cases}
        1 & x > 0 \\
        0 & x \leq 0
    \end{cases}
\end{align*}

\subsection{Ramp Clip (\texttt{ClippedReLU\_Ramp})}
Forward is the same hard clamp to $[0, C]$. The backward pass keeps slope $1$ up to
$C$, then decays \emph{linearly} to $0$ over $[C, rC]$ (ramp ratio $r$; $r=2$ in all
runs here, so the ramp ends at $2C$) -- this preserves the distinction between a
coordinate that is slightly over $C$ and one that is far over $C$, instead of treating
both identically:
\begin{align*}
    y &= \mathrm{clamp}(x, 0, C) \\
    \frac{\partial y}{\partial x} &=
    \begin{cases}
        1 & 0 < x \leq C \\
        \dfrac{rC - x}{rC - C} & C < x < rC \\
        0 & x \leq 0 \ \text{or}\ x \geq rC
    \end{cases}
\end{align*}

\subsection{Fixed Gradient-Norm Clip (\texttt{gradient\_clip\_val}$=21$)}
Also not an activation function -- a plain-ReLU CNN where each honest client's overall
gradient (all parameters concatenated) is rescaled, via
\texttt{torch.nn.utils.clip\_grad\_norm\_}, to a fixed global L2 norm threshold
$C_g = 21$, applied to the raw per-parameter \texttt{.grad} tensors right after
\texttt{loss.backward()} -- i.e.\ before flattening and before the momentum
accumulation described below:
\begin{align*}
    g &= \nabla_\theta \mathcal{L}(\theta) \quad \text{(concatenation of all parameter gradients)} \\
    g^{\text{clipped}} &=
    \begin{cases}
        g \cdot \dfrac{C_g}{\lVert g \rVert_2} & \lVert g \rVert_2 > C_g \\
        g & \text{otherwise}
    \end{cases}
\end{align*}
Unlike the adaptive client-side clip below, $C_g$ is a fixed constant (not
data-adaptive) and is not a quantile of anything.

\subsection{Adaptive Per-Coordinate Quantile Clip (\texttt{AdaptiveQuantileClip})}
Instead of a fixed constant $C$, the clip threshold is recomputed every forward pass as
the $\tau$-quantile of that layer's own activation coordinates (detached, i.e. treated
as a constant for backpropagation), so the threshold adapts to the current activation
distribution instead of being hand-picked. The backward pass uses the \emph{plain} clamp
derivative (hard zero above $C$):
\begin{align*}
    C &= \mathrm{quantile}_\tau\big(\mathrm{flatten}(x)\big) \quad \text{(detached)} \\
    y &= \mathrm{clamp}(x, 0, C) \\
    \frac{\partial y}{\partial x} &= \mathbb{1}[0 < x \leq C]
\end{align*}
Swept at $\tau \in \{0.80, 0.90\}$
(\emph{results pending -- not yet available at time of writing}). An STE-backward variant
was also implemented but dropped: the fixed-clip STE runs showed STE degrades even clean
(no-attack) accuracy on MNIST, so it was not worth sweeping.

\subsection{Adaptive Client-Side Gradient-Norm Clip}
Not an activation function -- a plain-ReLU CNN where each honest client clips the L2
norm of the (post-momentum) gradient vector it sends to the server, to the
$\tau$-quantile of its own last $W=100$ gradient norms (a sliding window, since
gradient norms are non-stationary and shrink over training). Let $v_t$ be the momentum
gradient at step $t$ and $n_t = \lVert v_t \rVert_2$; a history buffer of the last $W$
values of $n_t$ is maintained \emph{before} any clipping is applied, so the quantile
reflects the true, unclipped distribution:
\begin{align*}
    \tau\text{-threshold}_t &= \mathrm{quantile}_\tau\big(\{n_{t-W+1}, \dots, n_t\}\big) \\
    v_t^{\text{sent}} &=
    \begin{cases}
        v_t \cdot \dfrac{\tau\text{-threshold}_t}{n_t} & n_t > \tau\text{-threshold}_t \\
        v_t & \text{otherwise}
    \end{cases}
\end{align*}
The internal momentum buffer itself is never rescaled -- only the copy actually sent to
the server is clipped. Swept at $\tau \in \{0.70, 0.80\}$
(\emph{results pending -- not yet available at time of writing}).

\clearpage
""")

        # Everything mixed together in every grid -- no per-clip-value sectioning.
        f.write(r"\section{Best Overall Performance (Worst-Case Across Attacks)}" + "\n")
        items = [(best_test_path(c), c["caption"], f"best_overall_{c['key']}") for c in CONFIGS]
        f.write(make_grid(items, "0.32", "Best overall test accuracy (worst-case across attacks).",
                           "best_overall_grid", latex_dir=latex_dir))
        f.write("\n\\clearpage\n")

        f.write(r"\section{Best Aggregator under Specific Attacks}" + "\n\n")
        for attack in ATTACKS:
            f.write(f"\\subsection{{{attack['title']}}}\n")
            items = [(best_test_path(c, attack["tag"]), c["caption"], f"best_{attack['key']}_{c['key']}")
                     for c in CONFIGS]
            f.write(make_grid(items, "0.32", f"Best test accuracy under {attack['short']}.",
                               f"best_{attack['key']}_grid", latex_dir=latex_dir))
            f.write("\n\\clearpage\n")

        for agg in AGGREGATORS:
            f.write(f"\\section{{{agg['title']}}}\n")

            f.write(f"\\subsection{{{agg['short']} under Worst-Case across all Attacks}}\n")
            items = [(test_path(c, agg["tag"]), c["caption"], f"{agg['key']}_worst_case_{c['key']}")
                     for c in CONFIGS]
            f.write(make_grid(items, "0.32", f"{agg['title']} under Worst-Case across all Attacks.",
                               f"{agg['key']}_worst_case_grid", latex_dir=latex_dir))
            f.write("\n\\clearpage\n")

            for attack in ATTACKS:
                f.write(f"\\subsection{{{agg['short']} under {attack['short']}}}\n")
                items = [(test_path(c, agg["tag"], attack["tag"]), c["caption"],
                          f"{agg['key']}_{attack['key']}_{c['key']}") for c in CONFIGS]
                f.write(make_grid(items, "0.32", f"{agg['title']} under {attack['title']}.",
                                   f"{agg['key']}_{attack['key']}_grid", latex_dir=latex_dir))
                f.write("\n\\clearpage\n")

        f.write("\n\\end{document}\n")

    print(f"LaTeX file successfully written to {tex_path}")

    print("Compiling LaTeX report to PDF...")
    try:
        tex_filename = os.path.basename(tex_path)
        for _ in range(2):
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", tex_filename],
                cwd=latex_dir,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        print("LaTeX report compiled successfully: activation_clip_report.pdf")
    except Exception as e:
        print(f"[ERROR] Failed to compile LaTeX: {e}")


if __name__ == "__main__":
    main()
