#!/usr/bin/env python

from sxl.load import load_all_mouses
from sxl.analyses import (
    pearson_correlation_coefficient,
    evaluate_neurons_with_roc,
    plot_roc_curves_for_events,
)

df = load_all_mouses(
    {
        "F2_2": (
            "for_LJW/F2_2/df_f_zscore.npy.mat",
            [
                "for_LJW/F2_2/behavior1.xlsx",
                "for_LJW/F2_2/behavior2.xlsx",
                "for_LJW/F2_2/behavior3.xlsx",
                "for_LJW/F2_2/behavior4.xlsx",
                "for_LJW/F2_2/behavior5.xlsx",
            ],
        ),
        "F2_3": (
            "for_LJW/F2_3/df_f_zscore.npy.mat",
            [
                "for_LJW/F2_3/behavior1.xlsx",
                "for_LJW/F2_3/behavior2.xlsx",
                "for_LJW/F2_3/behavior3.xlsx",
                "for_LJW/F2_3/behavior4.xlsx",
                "for_LJW/F2_3/behavior5.xlsx",
            ],
        ),
    },
    event1_length=6299,
    event234_length=5749,
)

df_corr_events = pearson_correlation_coefficient(df, events=[1, 2, 3, 4, 5])

df_auroc = evaluate_neurons_with_roc(df, events=[1, 2, 3, 4, 5], permute_num=1000)
df_auroc["type"] = df_auroc["quantile"].transform(
    lambda quantile: (
        "excited"
        if quantile >= 0.95
        else "inhibited" if quantile <= 0.05 else "non-responsive"
    ),
    axis=0,
)

print(df_auroc.groupby(["mouse", "event"])["type"].value_counts())

fig = plot_roc_curves_for_events(df_auroc)
fig.savefig("for_LJW/auroc_analysis.pdf")
