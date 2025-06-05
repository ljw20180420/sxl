import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import scipy.io as sio
import os
import pandas as pd
import pathlib
import multiprocessing
import itertools
from tqdm import tqdm


# 读取每个小鼠的神经元数据和行为事件数据
def load_mouse_data(base_folder, mouse_folder, behavior_file):
    # 读取神经元数据
    mat_file_path = os.path.join(
        base_folder, mouse_folder, "suite2p", "plane0", "df_f_zscore.npy.mat"
    )
    mat_data = sio.loadmat(mat_file_path)
    neuron_data = mat_data["df_f_zscore"]  # 假设数据存储在键 'df_f_zscore' 下

    # 读取行为事件数据
    excel_file_path = os.path.join(base_folder, mouse_folder, behavior_file)
    behavior_data = pd.read_excel(excel_file_path)

    return neuron_data, behavior_data


def do_random_permute_auroc(i, labels, neuron_activity):
    fpr, tpr, _ = roc_curve(labels, np.random.permutation(neuron_activity))
    return auc(fpr, tpr).item()


# 使用ROC分析评估神经元对行为事件的响应
def evaluate_neurons_with_roc(
    neuron_data, behavior_data, event1_length=6299, event234_length=5749
):
    _, n_frames = neuron_data.shape
    aurocs, fprs, tprs, quantiles = [], [], [], []

    # 创建行为事件标签
    assert behavior_data.columns[0].lower().startswith(
        "start"
    ) and behavior_data.columns[1].lower().startswith(
        "end"
    ), "the first two columns of behavior file must be start and end, and named S(s)tart# and E(e)nd#, where # = 1, 2, 3, 4, 5"

    labels = np.full(n_frames, False)
    for start, end in zip(behavior_data.iloc[:, 0], behavior_data.iloc[:, 1]):
        if np.isnan(start) or np.isnan(end):
            print(f"Warning: NaN value encountered in event. Skipping this event.")
            continue
        start = int(start) - 1
        end = int(end)
        if start < 0:
            start = 0
        if end > n_frames:
            end = n_frames
        labels[start:end] = True

    if behavior_data.columns[0].lower() == "start1":
        labels = labels[:event1_length]
        neuron_data = neuron_data[:, :event1_length]
    elif behavior_data.columns[0].lower() == "start5":
        labels = labels[event1_length + event234_length :]
        neuron_data = neuron_data[:, event1_length + event234_length :]
    else:
        labels = labels[event1_length : event1_length + event234_length]
        neuron_data = neuron_data[:, event1_length : event1_length + event234_length]

    # 对每个神经元进行ROC分析
    for neuron_activity in tqdm(neuron_data):
        # 计算ROC曲线和AUROC值
        fpr, tpr, _ = roc_curve(labels, neuron_activity)
        auroc = auc(fpr, tpr)
        aurocs.append(auroc.item())
        fprs.append(fpr)
        tprs.append(tpr)

        with multiprocessing.Pool() as pool:
            random_permute_aurocs = pool.starmap(
                do_random_permute_auroc,
                zip(
                    range(1000),
                    itertools.repeat(labels),
                    itertools.repeat(neuron_activity),
                ),
            )

        random_permute_aurocs = np.array(random_permute_aurocs)
        quantile = sum(random_permute_aurocs < auroc) / len(random_permute_aurocs)
        quantiles.append(quantile.item())

    return (
        pd.DataFrame(
            {
                "auroc": aurocs,
                "fpr": fprs,
                "tpr": tprs,
                "quantile": quantiles,
                "neuron activity": list(neuron_data),
            }
        ),
        labels,
    )


# 绘制ROC曲线（只针对事件1）
def plot_roc_curves_for_event1(df):
    plt.figure(figsize=(10, 8))
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

        plt.plot(fpr, tpr, color=color, lw=2, label=f"{label} (auROC = {auroc:.2f})")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Neurons Responsive to Event 2")
    plt.legend(loc="lower right")
    plt.tight_layout()
    return plt


# 主函数
if __name__ == "__main__":
    # 设置基础文件夹和小鼠文件夹（共11只小鼠）
    base_folder = pathlib.Path(
        "/media/ljw/沈秀莲/OFC/OFC_EXPERIMENT/TP/analyse_data/multiple_days2"
    )
    mouse_folders = ["F2_10"]
    # base_folder = pathlib.Path(r"F:\OFC\OFC_EXPERIMENT\TP\analyse_data\multiple_days2")
    # mouse_folders = ["F2_10", "UF2_2", "UF2_3", "UF2_4"]
    behavior_file = "behavior1.xlsx"

    counts = {
        "excited": [],
        "inhibited": [],
        "nonresponse": [],
    }
    # 处理每只小鼠的数据
    for mouse_folder in mouse_folders:
        # 读取小鼠的神经元数据和行为事件数据
        neuron_data, behavior_data = load_mouse_data(
            base_folder, mouse_folder, behavior_file
        )

        # 使用ROC分析评估神经元对事件1的响应
        df, labels = evaluate_neurons_with_roc(
            neuron_data, behavior_data, event1_length=6299, event234_length=5749
        )

        # 保存为 .mat 文件
        category_folder = base_folder / "neuron_activity_data" / mouse_folder
        os.makedirs(category_folder, exist_ok=True)

        for type in ["excited", "inhibited", "nonresponse"]:
            if type == "excited":
                df_type = df.query("quantile >= 0.95")
            elif type == "inhibited":
                df_type = df.query("quantile <= 0.05")
            else:
                df_type = df.query("quantile < 0.95 & quantile > 0.05")
            sio.savemat(
                category_folder / f"{type}_neuron_activity.mat",
                {
                    column: np.stack(df_type[column])
                    for column in ["auroc", "quantile", "neuron activity"]
                },
            )
            counts[type].append(len(df_type))

    # 计算总数
    breakpoint()
    for category, count in counts.items():
        total_neurons = sum(count)
        for mouse, ct in zip(mouse_folders, count):
            percentage = (ct / total_neurons * 100) if total_neurons > 0 else 0
            print(f"{category} {mouse}: {ct} ({percentage:.2f}%)")

    # 绘制ROC曲线
    plt = plot_roc_curves_for_event1(df)

    # 保存图形到PDF文件
    plt.savefig(base_folder / "event2_analysis.pdf")
    plt.close()
