import pandas as pd
import scipy.io as sio
import numpy as np


def load_all_mouses(mouse_data, event1_length=3299, event234_length=5749):
    dfs = []
    for mouse_name, file_pair in mouse_data.items():
        z_score_file, behavior_files = file_pair
        # 读取神经元数据, 数据存储在键 'df_f_zscore' 下
        neuron_data = sio.loadmat(z_score_file)["df_f_zscore"]
        n_cells, n_frames = neuron_data.shape
        cell_names = [f"cell{idx_cell}" for idx_cell in range(n_cells)]
        df = pd.DataFrame(neuron_data.T, columns=cell_names)
        df.insert(0, "mouse", mouse_name)
        df["time"] = df.index
        names = ["mouse", "time"]
        for behavior_file in behavior_files:
            # 读取行为事件数据
            behavior_data = pd.read_excel(behavior_file)
            events = np.full(n_frames, False)
            if behavior_data.columns[0].lower() == "start1":
                events[:event1_length] = True
            elif behavior_data.columns[0].lower() == "start5":
                events[event1_length + event234_length :] = True
            else:
                events[event1_length : event1_length + event234_length] = True
            labels = np.full(n_frames, False)
            for start, end in zip(behavior_data["Start5"], behavior_data["End5"]):
                if np.isnan(start) or np.isnan(end):
                    print(
                        f"Warning: NaN value encountered in event. Skipping this event."
                    )
                    continue
                start = int(start) - 1
                end = int(end)
                if start < 0:
                    start = 0
                if end > n_frames:
                    end = n_frames
                labels[start:end] = True
            event_name = "event" + behavior_data.columns[0].lower()[-1]
            label_name = "label" + behavior_data.columns[0].lower()[-1]
            df[event_name] = events
            df[label_name] = labels
            names += [event_name, label_name]

        df = pd.melt(
            df,
            id_vars=names,
            value_vars=cell_names,
            var_name="cell",
            value_name="signal",
        )
        df.insert(1, "cell", df.pop("cell"))

        dfs.append(df)

    return pd.concat(dfs).reset_index(drop=True)
