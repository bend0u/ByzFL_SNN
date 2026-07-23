"""STEP 0 (read-only): generate empty-accuracy template CSVs for the models
whose local results/ tree does NOT contain the complete (aggregator x attack x
f x gamma) grid needed to match the "mixed report" (CC, GM, MK, TrMean; 3
attacks; f=0..5; gamma in {1.0,0.66,0.33,0.0}). To be filled by hand from the
PDF. No training, no result files are read here -- just grid enumeration.
"""
import os
import pandas as pd

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

AGGREGATORS = ["TrMean", "GeometricMedian", "MultiKrum", "CenteredClipping"]
ATTACKS = ["Optimal_ALittleIsEnough_neg1", "SignFlipping", "Optimal_InnerProductManipulation"]
GAMMAS = [1.0, 0.66, 0.33, 0.0]
F_VALUES = [0, 1, 2, 3, 4, 5]


def make_template(model_name):
    rows = []
    for gamma in GAMMAS:
        for f in F_VALUES:
            for agg in AGGREGATORS:
                for attack in ATTACKS:
                    rows.append(dict(model=model_name, gamma=gamma, f=f, aggregator=agg, attack=attack, accuracy=""))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    for model_name in ["snn_atan12", "cnn_relu"]:
        df = make_template(model_name)
        path = os.path.join(OUT_DIR, f"robustness_template_{model_name}.csv")
        df.to_csv(path, index=False)
        print(f"Wrote {path} ({len(df)} rows to fill)")
