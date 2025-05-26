#!/usr/bin/env python

from sxl.load import load_all_mouses
from sxl.analyses import (
    pearson_correlation_coefficient,
    evaluate_neurons_with_roc,
    plot_roc_curves_for_events,
)

df = load_all_mouses(
    {
        "F1_3": (
            "data/mouse/suite2p/plane0/df_f_zscore.npy.mat",
            ["data/mouse/behavior_data5.xlsx"],
        ),
        "F1_fake": (
            "data/mouse/suite2p/plane0/df_f_zscore.npy.mat",
            ["data/mouse/behavior_data5.xlsx"],
        ),
    },
    event1_length=3299,
    event234_length=5749,
)

# df_corr_events = pearson_correlation_coefficient(df, events=[5])

df_auroc = evaluate_neurons_with_roc(df, events=[5], permute_num=1000)
df_auroc["type"] = df_auroc["quantile"].transform(
    lambda quantile: (
        "excited"
        if quantile >= 0.95
        else "inhibited" if quantile <= 0.05 else "non-responsive"
    ),
    axis=0,
)

print(df_auroc.groupby("mouse")["type"].value_counts())

fig = plot_roc_curves_for_events(df_auroc)
fig.savefig("data/event1_analysis.pdf")
