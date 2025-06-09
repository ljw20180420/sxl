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
        df.reset_index(level=["event"] + [f"label{event}" for event in events])
        .groupby(["mouse", "time"])
        .agg(
            {
                "event": "first",
                **{f"label{event}": "first" for event in events},
                "signal": "mean",
            }
        )
    )

    dfs = []
    for event in events:
        if event == 1:
            ev_filter = "event1"
        elif event == 5:
            ev_filter = "event5"
        else:
            ev_filter = "event234"
        dfs.append(
            df_mean_signal.query("event == @ev_filter")
            .groupby("mouse")
            .apply(
                lambda dfg: pd.Series(
                    (
                        f"label{event}",
                        dfg[f"label{event}"].corr(dfg["signal"]),
                        dfg[f"label{event}"]
                        .reset_index(drop=True)
                        .corr(pd.Series(np.random.permutation(dfg["signal"]))),
                    ),
                    index=["label", "corr", "shuffle"],
                )
            )
        )

    return pd.concat(dfs).set_index("label", append=True).sort_index()


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

    print(
        {
            "event": event,
            "mouse": dfg.index.get_level_values("mouse")[0],
            "cell": dfg.index.get_level_values("cell")[0],
        }
    )

    return pd.Series(
        (f"label{event}", auroc, fpr, tpr, quantile),
        index=["label", "auroc", "fpr", "tpr", "quantile"],
    )


# 使用ROC分析评估神经元对行为事件的响应
def evaluate_neurons_with_roc(
    df: pd.DataFrame, events: list = [1, 2, 3, 4, 5], permute_num=1000
):
    dfs = []
    for event in events:
        if event == 1:
            ev_filter = "event1"
        elif event == 5:
            ev_filter = "event5"
        else:
            ev_filter = "event234"
        dfs.append(
            df.reset_index(level=[f"label{event}" for event in events])
            .query("event == @ev_filter")
            .groupby(["mouse", "cell"])
            .apply(calculate_auroc_etc, event=event, permute_num=permute_num)
        )

    return pd.concat(dfs).set_index("label", append=True).sort_index()


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
        fpr = df.iloc[idx]["fpr"]
        tpr = df.iloc[idx]["tpr"]
        auroc = df.iloc[idx]["auroc"]

        ax.plot(fpr, tpr, color=color, lw=2, label=f"{label} (auROC = {auroc:.2f})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves for Neurons Responsive to Event 1")
    fig.legend(loc="lower right")
    fig.tight_layout()
    return fig
