import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
import scipy.io as sio
import os
from multiprocessing import Pool, cpu_count
import pathlib


# 加载数据函数
def load_neuron_data(base_folder, event_folder, mouse_folder):
    excited_file = os.path.join(
        base_folder, event_folder, mouse_folder, "excited_neuron_activity.mat"
    )
    inhibited_file = os.path.join(
        base_folder, event_folder, mouse_folder, "inhibited_neuron_activity.mat"
    )
    nonresponse_file = os.path.join(
        base_folder, event_folder, mouse_folder, "nonresponse_neuron_activity.mat"
    )

    if not (
        os.path.exists(excited_file)
        and os.path.exists(inhibited_file)
        and os.path.exists(nonresponse_file)
    ):
        raise FileNotFoundError(
            f"One or more data files not found in folder: {mouse_folder}"
        )

    excited_data = sio.loadmat(excited_file)
    inhibited_data = sio.loadmat(inhibited_file)
    nonresponse_data = sio.loadmat(nonresponse_file)

    # 提取 neuron_activity 参数（注意字段名称中的空格）
    excited_neurons = excited_data.get("neuron activity", np.array([]))
    inhibited_neurons = inhibited_data.get("neuron activity", np.array([]))
    nonresponse_neurons = nonresponse_data.get("neuron activity", np.array([]))

    return excited_neurons, inhibited_neurons, nonresponse_neurons


# 辅助函数：找出共同的神经元
def find_common_neurons(neurons1, neurons2):
    common_neurons = []
    for neuron1 in neurons1:
        for neuron2 in neurons2:
            if neuron1.shape == neuron2.shape and np.array_equal(neuron1, neuron2):
                common_neurons.append(neuron1)
    return common_neurons


# 主函数
def main():
    base_folder = pathlib.Path(
        "/media/ljw/沈秀莲/OFC/OFC_EXPERIMENT/TP/analyse_data/multiple_days1"
    )
    # base_folder = r"F:\OFC\OFC_EXPERIMENT\TP\analyse_data\multiple_days1\ROC_20250516"
    event_folders = [
        "event_2_neuron_activity_data",
        "event_3_neuron_activity_data",
        "event_4_neuron_activity_data",
    ]
    mouse_folders = ["F1_3", "UF1_1", "UF1_3", "UF1_5", "UF1_6", "UF1_8", "UF1_10"]

    # 用于存储所有小鼠的响应和不响应神经元
    event2_responsive_neurons = []
    event3_responsive_neurons = []
    event4_responsive_neurons = []
    all_neurons = set()

    for event_folder in event_folders:
        for mouse_folder in mouse_folders:
            try:
                excited_neurons, inhibited_neurons, nonresponse_neurons = (
                    load_neuron_data(base_folder, event_folder, mouse_folder)
                )

                # 假设 neuron_activity 的行数表示神经元数量
                excited_count = (
                    excited_neurons.shape[0] if excited_neurons.size > 0 else 0
                )
                inhibited_count = (
                    inhibited_neurons.shape[0] if inhibited_neurons.size > 0 else 0
                )
                nonresponse_count = (
                    nonresponse_neurons.shape[0] if nonresponse_neurons.size > 0 else 0
                )

                # 更新响应神经元
                if event_folder == "event_2_neuron_activity_data":
                    event2_responsive_neurons.append(excited_neurons)
                    event2_responsive_neurons.append(inhibited_neurons)
                elif event_folder == "event_3_neuron_activity_data":
                    event3_responsive_neurons.append(excited_neurons)
                    event3_responsive_neurons.append(inhibited_neurons)
                elif event_folder == "event_4_neuron_activity_data":
                    event4_responsive_neurons.append(excited_neurons)
                    event4_responsive_neurons.append(inhibited_neurons)

                # 更新所有神经元集合
                neuron_count = excited_count + inhibited_count + nonresponse_count
                all_neurons.update(range(neuron_count))

                print(f"Event folder {event_folder}, Mouse folder {mouse_folder}:")
                print(f"  Responsive neurons: {excited_count + inhibited_count}")
                print(f"  Non-responsive neurons: {nonresponse_count}")
                print(f"  Total neurons: {neuron_count}\n")

            except FileNotFoundError as e:
                print(
                    f"Error loading data for event folder {event_folder}, mouse folder {mouse_folder}: {e}"
                )
                continue

    # 合并所有小鼠的响应神经元数据
    event2_responsive_combined = (
        np.vstack(event2_responsive_neurons)
        if event2_responsive_neurons
        else np.array([])
    )
    event3_responsive_combined = (
        np.vstack(event3_responsive_neurons)
        if event3_responsive_neurons
        else np.array([])
    )
    event4_responsive_combined = (
        np.vstack(event4_responsive_neurons)
        if event4_responsive_neurons
        else np.array([])
    )

    # 使用多进程计算共同神经元
    with Pool(cpu_count()) as pool:
        # 找出对事件2和事件3均响应的神经元
        common_neurons_event2_event3 = pool.starmap(
            find_common_neurons,
            [(event2_responsive_combined, event3_responsive_combined)],
        )[0]

        # 找出对事件2和事件4均响应的神经元
        common_neurons_event2_event4 = pool.starmap(
            find_common_neurons,
            [(event2_responsive_combined, event4_responsive_combined)],
        )[0]

        # 找出对事件3和事件4均响应的神经元
        common_neurons_event3_event4 = pool.starmap(
            find_common_neurons,
            [(event3_responsive_combined, event4_responsive_combined)],
        )[0]

        # 找出对事件2、事件3和事件4均响应的神经元
        common_neurons_event2_event3_event4 = pool.starmap(
            find_common_neurons,
            [(common_neurons_event2_event3, event4_responsive_combined)],
        )[0]

    # 找出仅对事件2响应的神经元
    only_event2 = []
    for neuron2 in event2_responsive_combined:
        found_in_event3 = False
        found_in_event4 = False
        for neuron3 in event3_responsive_combined:
            if neuron2.shape == neuron3.shape and np.array_equal(neuron2, neuron3):
                found_in_event3 = True
                break
        for neuron4 in event4_responsive_combined:
            if neuron2.shape == neuron4.shape and np.array_equal(neuron2, neuron4):
                found_in_event4 = True
                break
        if not found_in_event3 and not found_in_event4:
            only_event2.append(neuron2)
    only_event2 = np.array(only_event2)

    # 找出仅对事件3响应的神经元
    only_event3 = []
    for neuron3 in event3_responsive_combined:
        found_in_event2 = False
        found_in_event4 = False
        for neuron2 in event2_responsive_combined:
            if neuron3.shape == neuron2.shape and np.array_equal(neuron3, neuron2):
                found_in_event2 = True
                break
        for neuron4 in event4_responsive_combined:
            if neuron3.shape == neuron4.shape and np.array_equal(neuron3, neuron4):
                found_in_event4 = True
                break
        if not found_in_event2 and not found_in_event4:
            only_event3.append(neuron3)
    only_event3 = np.array(only_event3)

    # 找出仅对事件4响应的神经元
    only_event4 = []
    for neuron4 in event4_responsive_combined:
        found_in_event2 = False
        found_in_event3 = False
        for neuron2 in event2_responsive_combined:
            if neuron4.shape == neuron2.shape and np.array_equal(neuron4, neuron2):
                found_in_event2 = True
                break
        for neuron3 in event3_responsive_combined:
            if neuron4.shape == neuron3.shape and np.array_equal(neuron4, neuron3):
                found_in_event3 = True
                break
        if not found_in_event2 and not found_in_event3:
            only_event4.append(neuron4)
    only_event4 = np.array(only_event4)

    # 找出对事件2和事件3响应但不对事件4响应的神经元
    event2_event3_only = []
    for neuron in common_neurons_event2_event3:
        found_in_event4 = False
        for neuron4 in event4_responsive_combined:
            if neuron.shape == neuron4.shape and np.array_equal(neuron, neuron4):
                found_in_event4 = True
                break
        if not found_in_event4:
            event2_event3_only.append(neuron)
    event2_event3_only = np.array(event2_event3_only)

    # 找出对事件3和事件4响应但不对事件2响应的神经元
    event3_event4_only = []
    for neuron in common_neurons_event3_event4:
        found_in_event2 = False
        for neuron2 in event2_responsive_combined:
            if neuron.shape == neuron2.shape and np.array_equal(neuron, neuron2):
                found_in_event2 = True
                break
        if not found_in_event2:
            event3_event4_only.append(neuron)
    event3_event4_only = np.array(event3_event4_only)

    # 找出对事件2和事件4响应但不对事件3响应的神经元
    event2_event4_only = []
    for neuron in common_neurons_event2_event4:
        found_in_event3 = False
        for neuron3 in event3_responsive_combined:
            if neuron.shape == neuron3.shape and np.array_equal(neuron, neuron3):
                found_in_event3 = True
                break
        if not found_in_event3:
            event2_event4_only.append(neuron)
    event2_event4_only = np.array(event2_event4_only)

    # 找出对所有三个事件都响应的神经元
    all_events = common_neurons_event2_event3_event4

    # 找出对任何事件都不响应的神经元
    nonresponsive_neurons = (
        all_neurons
        - set(range(len(event2_responsive_combined)))
        - set(range(len(event3_responsive_combined)))
        - set(range(len(event4_responsive_combined)))
    )

    # 打印分类结果
    print("Classification Results:")
    print(f"Only Event2: {len(only_event2)}")
    print(f"Only Event3: {len(only_event3)}")
    print(f"Only Event4: {len(only_event4)}")
    print(f"Event2 & Event3 only: {len(event2_event3_only)}")
    print(f"Event3 & Event4 only: {len(event3_event4_only)}")
    print(f"Event2 & Event4 only: {len(event2_event4_only)}")
    print(f"All Events: {len(all_events)}")
    print(f"Non-responsive to all events: {len(nonresponsive_neurons)}")

    # 创建 Venn 图
    venn_diagram = venn3(
        subsets=(
            len(only_event2),
            len(only_event3),
            len(event2_event3_only),
            len(only_event4),
            len(event2_event4_only),
            len(event3_event4_only),
            len(all_events),
        ),
        set_labels=("事件2", "事件3", "事件4"),
    )

    # 计算并显示非响应神经元的数量和百分比
    total_neurons = len(all_neurons)
    if total_neurons > 0:
        nonresponsive_percentage = len(nonresponsive_neurons) / total_neurons * 100
        plt.text(
            0.5,
            -0.1,
            f"非响应神经元: {len(nonresponsive_neurons)} ({nonresponsive_percentage:.1f}%)",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )

    plt.title("神经元对三个事件的响应情况")
    plt.show()


if __name__ == "__main__":
    main()
