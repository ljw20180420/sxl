import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import multiprocessing
import itertools


def pearson_correlation_coefficient(
    df: pd.DataFrame, events: list = [1, 2, 3, 4, 5]
) -> pd.DataFrame:
    df_mean_signal = (
        df.groupby(["mouse", "time"])
        .agg(
            {
                **{f"event{event}": "first" for event in events},
                **{f"label{event}": "first" for event in events},
                "signal": "mean",
            }
        )
        .reset_index(drop=False)
    )

    return pd.concat(
        [
            df_mean_signal.query(f"event{event}")
            .groupby("mouse")
            .apply(
                lambda dfg: pd.Series(
                    (
                        dfg[f"label{event}"].corr(dfg["signal"]),
                        dfg[f"label{event}"]
                        .reset_index(drop=True)
                        .corr(pd.Series(np.random.permutation(dfg["signal"]))),
                        f"event{event}",
                    ),
                    index=["corr", "shuffle", "event"],
                )
            )
            .reset_index(drop=False)
            for event in events
        ]
    ).reset_index(drop=True)


def do_random_permute_auroc(i, labels, signals):
    fpr, tpr, _ = roc_curve(labels, np.random.permutation(signals))
    return auc(fpr, tpr).item()


def calculate_auroc_etc(dfg, event, permute_num=1000):
    fpr, tpr, _ = roc_curve(dfg[f"label{event}"], dfg["signal"])
    auroc = auc(fpr, tpr).item()
    with multiprocessing.Pool() as pool:
        random_permute_aurocs = pool.starmap(
            do_random_permute_auroc,
            zip(
                range(0, permute_num),
                itertools.repeat(dfg[f"label{event}"]),
                itertools.repeat(dfg["signal"]),
            ),
        )
    quantile = sum(np.array(random_permute_aurocs) < auroc) / len(random_permute_aurocs)

    return pd.Series(
        (f"event{event}", auroc, fpr, tpr, quantile),
        index=["event", "auroc", "fpr", "tpr", "quantile"],
    )


# 使用ROC分析评估神经元对行为事件的响应
def evaluate_neurons_with_roc(
    df: pd.DataFrame, events: list = [1, 2, 3, 4, 5], permute_num=1000
):
    return pd.concat(
        [
            df.query(f"event{event}")
            .groupby(["mouse", "cell"])
            .apply(calculate_auroc_etc, event=event, permute_num=permute_num)
            .reset_index(drop=False)
            for event in events
        ]
    ).reset_index(drop=True)


# 绘制ROC曲线（只针对事件1）
def plot_roc_curves_for_events(df: pd.DataFrame):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
    for idx, color, label in zip(
        [
            df["quantile"].argmax(),
            df["quantile"].argmin(),
            (df["quantile"] - 0.5).abs().argmin(),
        ],
        ["orchid", "red", "gainsboro"],
        ["excited", "inhibited", "Non-responsive"],
    ):
        fpr = df.loc[idx, "fpr"]
        tpr = df.loc[idx, "tpr"]
        auroc = df.loc[idx, "auroc"]

        ax.plot(fpr, tpr, color=color, lw=2, label=f"{label} (auROC = {auroc:.2f})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves for Neurons Responsive to Event 1")
    fig.legend(loc="lower right")
    fig.tight_layout()
    return fig
