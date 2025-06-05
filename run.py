#!/usr/bin/env python

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
            # "F2_3": (
            #     "for_LJW/F2_3/df_f_zscore.npy.mat",
            #     [
            #         "for_LJW/F2_3/behavior1.xlsx",
            #         "for_LJW/F2_3/behavior2.xlsx",
            #         "for_LJW/F2_3/behavior3.xlsx",
            #         "for_LJW/F2_3/behavior4.xlsx",
            #         "for_LJW/F2_3/behavior5.xlsx",
            #     ],
            # ),
        },
        event1_length=6299,
        event234_length=5749,
    )

    df.to_csv("for_LJW/data.csv", index=False)

    df_corr_events = pearson_correlation_coefficient(df, events=[1, 2, 3, 4, 5])
    df_corr_events.to_csv("for_LJW/corr.csv", index=False)

    df_auroc = evaluate_neurons_with_roc(df, events=[1, 2, 3, 4, 5], permute_num=1000)
    df_auroc["type"] = df_auroc["quantile"].transform(
        lambda quantile: (
            "excited"
            if quantile >= 0.95
            else "inhibited" if quantile <= 0.05 else "non-responsive"
        ),
        axis=0,
    )

    df_auroc.loc[:, ["mouse", "cell", "event", "auroc", "quantile", "type"]].to_csv(
        "for_LJW/auroc_analysis.csv", index=False
    )

    df_auroc.groupby(["mouse", "event"])["type"].value_counts().reset_index(
        drop=False
    ).to_csv("for_LJW/auroc_analysis_count.csv", index=False)

    fig = plot_roc_curves_for_events(df_auroc)
    fig.savefig("for_LJW/auroc_analysis.pdf")

    df_auroc_pivot = (
        df_auroc.loc[:, ["mouse", "cell", "event", "type"]]
        .pivot(columns="event", index=["mouse", "cell"], values="type")
        .reset_index(drop=False)
    )

    for event in ["event2", "event3", "event4"]:
        df_auroc_pivot[event] = (
            df_auroc_pivot[event]
            .str.replace("excited", "response")
            .replace("inhibited", "response")
        )

    df_auroc_pivot.loc[:, ["event2", "event3", "event4"]].value_counts().reset_index(
        drop=False
    ).to_csv("for_LJW/venn.csv", index=False)
