import pandas as pd
import scipy.io as sio
import numpy as np
import re


def load_all_mouses(mouse_data, event1_length=3299, event234_length=5749):
    dfs = []
    for mouse_name, file_pair in mouse_data.items():
        z_score_file, behavior_files = file_pair
        # 读取神经元数据, 数据存储在键 'df_f_zscore' 下
        neuron_data = sio.loadmat(z_score_file)["df_f_zscore"]
        n_cells, n_frames = neuron_data.shape
        cell_names = [f"cell{idx_cell}" for idx_cell in range(n_cells)]
        time_indices = {
            "mouse": [mouse_name] * n_frames,
            "time": range(n_frames),
            "event": ["event1"] * event1_length
            + ["event234"] * event234_length
            + ["event5"] * (n_frames - event1_length - event234_length),
        }
        for behavior_file in behavior_files:
            # 读取行为事件数据
            behavior_data = pd.read_excel(behavior_file)
            assert behavior_data.columns[0].lower().startswith(
                "start"
            ) and behavior_data.columns[1].lower().startswith(
                "end"
            ), "the first two columns of behavior file must be start and end, and named S(s)tart# and E(e)nd#, where # = 1, 2, 3, 4, 5"
            labels = np.full(n_frames, False)
            for start, end in zip(behavior_data.iloc[:, 0], behavior_data.iloc[:, 1]):
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
            event_tail_num = re.search(
                r"start(\d+)", behavior_data.columns[0], re.IGNORECASE
            ).group(1)
            time_indices["label" + event_tail_num] = labels

        multi_index = pd.DataFrame(time_indices)
        multi_index["event"] = multi_index["event"].astype("category")
        df = pd.DataFrame(
            data=neuron_data.T,
            index=pd.MultiIndex.from_frame(multi_index),
            columns=cell_names,
        )
        df = pd.melt(
            df,
            value_vars=cell_names,
            var_name="cell",
            value_name="signal",
            ignore_index=False,
        )
        df["cell"] = pd.Categorical(df["cell"])
        df = df.set_index("cell", append=True).sort_index()

        dfs.append(df)

    return pd.concat(dfs)
