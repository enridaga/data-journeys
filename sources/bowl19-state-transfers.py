
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
dtypes = {
        "title": "category",
        "event_id": "category",
        "event_count": "int16",
        "event_code": "int16",
        "game_time": "int32",
        "title": "category",
        "type": "category",
        "world": "category",
        "installation_id": "category",
        "game_session": "category",
    }

D = pd.read_csv("/kaggle/input/data-science-bowl-2019/train.csv", dtype=dtypes, parse_dates=["timestamp"])
dd=D[["installation_id", "game_session", "title", "world"]].drop_duplicates()
dd["prev_title"]=dd.groupby("installation_id")["title"].shift()

dd["idx"]=dd.groupby("installation_id").cumcount()

order=dd[["world", "title"]].drop_duplicates()
order=order.merge(dd.groupby("title")["idx"].mean().rename("mean_title").reset_index())
order=order.merge(dd.groupby("world")["idx"].mean().rename("mean_world").reset_index())
order=order.sort_values(["mean_world", "mean_title"])

title_map=D[["world", "type", "title"]].drop_duplicates().set_index("title")
plt.figure(figsize=(15, 15))
labels=order["title"].map(lambda x:f"{x} - {title_map.loc[x, 'world']}")
plot_data=dd.groupby(["prev_title", "title"]).size().unstack("title").fillna(0).reindex(index=order["title"], columns=order["title"])
sns.heatmap(plot_data, cbar=False, xticklabels=labels, yticklabels=labels, cmap="plasma", robust=True);