#!/usr/bin/env python

import pandas as pd
from sxl.load import load_all_mouses
from sxl.analyses import (
    pearson_correlation_coefficient,
    evaluate_neurons_with_roc,
    plot_roc_curves_for_events,
)

if __name__ == "__main__":
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
    breakpoint()
    df.to_csv("for_LJW/data.csv")

    df_corr_events = pearson_correlation_coefficient(df, events=[1, 2, 3, 4, 5])
    df_corr_events.to_csv("for_LJW/corr.csv")

    df_auroc = evaluate_neurons_with_roc(df, events=[1, 2, 3, 4, 5], permute_num=1000)
    df_auroc["type"] = df_auroc["quantile"].transform(
        lambda quantile: (
            "excited"
            if quantile >= 0.95
            else "inhibited" if quantile <= 0.05 else "non-responsive"
        ),
        axis=0,
    )

    df_auroc.loc[:, ["auroc", "quantile", "type"]].to_csv("for_LJW/auroc_analysis.csv")

    df_auroc.groupby(["mouse", "label"])["type"].value_counts().to_csv(
        "for_LJW/auroc_analysis_count.csv"
    )

    fig = plot_roc_curves_for_events(df_auroc)
    fig.savefig("for_LJW/auroc_analysis.pdf")

    df_auroc_pivot = (
        df_auroc.loc[:, ["type"]]
        .reset_index(level="label")
        .pivot(columns="label", values="type")
    )

    for label in ["label2", "label3", "label4"]:
        df_auroc_pivot[label] = (
            df_auroc_pivot[label]
            .str.replace("excited", "response")
            .replace("inhibited", "response")
        )

    df_auroc_pivot.loc[:, ["label2", "label3", "label4"]].value_counts().to_csv(
        "for_LJW/venn.csv"
    )
