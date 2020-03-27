
import numpy as np
import pandas as pd
import os
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.patches as patches
pd.set_option('max_columns', 100)

data_path = "/kaggle/input/nfl-big-data-bowl-2020/"
train_df = pd.read_csv(data_path + "train.csv", low_memory=False)
train_df.head()
print(f"The total number of games in the training data is {train_df['GameId'].nunique()}")
print(f"The total number of plays in the training data is {train_df['PlayId'].nunique()}")
print(f"The NFL seasons in the training data are {train_df['Season'].unique().tolist()}")
from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.express as px

temp_df = train_df.query("NflIdRusher == NflId")
fig = px.histogram(temp_df, x="Yards")

layout = go.Layout(
    title=go.layout.Title(
        text="Distribution of Yards (Target)",
        x=0.5
    ),
    font=dict(size=14),
    width=800,
    height=500,
)
fig.update_layout(layout)
fig.show()
def create_football_field(linenumbers=True,
                          endzones=True,
                          highlight_line=False,
                          highlight_line_number=50,
                          highlighted_name='Line of Scrimmage',
                          fifty_is_los=False,
                          figsize=(12*2, 6.33*2)):
    """
    Function that plots the football field for viewing plays.
    Allows for showing or hiding endzones.
    """
    rect = patches.Rectangle((0, 0), 120, 53.3, linewidth=0.1,
                             edgecolor='r', facecolor='darkgreen', zorder=0)

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.add_patch(rect)

    plt.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
              80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
             [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
              53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
             color='white')
    if fifty_is_los:
        plt.plot([60, 60], [0, 53.3], color='gold')
        plt.text(62, 50, '<- Player Yardline at Snap', color='gold')
    # Endzones
    if endzones:
        ez1 = patches.Rectangle((0, 0), 10, 53.3,
                                linewidth=0.1,
                                edgecolor='r',
                                facecolor='blue',
                                alpha=0.2,
                                zorder=0)
        ez2 = patches.Rectangle((110, 0), 120, 53.3,
                                linewidth=0.1,
                                edgecolor='r',
                                facecolor='blue',
                                alpha=0.2,
                                zorder=0)
        ax.add_patch(ez1)
        ax.add_patch(ez2)
    plt.xlim(0, 120)
    plt.ylim(-5, 58.3)
    plt.axis('off')
    if linenumbers:
        for x in range(20, 110, 10):
            numb = x
            if x > 50:
                numb = 120 - x
            plt.text(x, 5, str(numb - 10),
                     horizontalalignment='center',
                     fontsize=20,  # fontname='Arial',
                     color='white')
            plt.text(x - 0.95, 53.3 - 5, str(numb - 10),
                     horizontalalignment='center',
                     fontsize=20,  # fontname='Arial',
                     color='white', rotation=180)
    if endzones:
        hash_range = range(11, 110)
    else:
        hash_range = range(1, 120)

    for x in hash_range:
        ax.plot([x, x], [0.4, 0.7], color='white')
        ax.plot([x, x], [53.0, 52.5], color='white')
        ax.plot([x, x], [22.91, 23.57], color='white')
        ax.plot([x, x], [29.73, 30.39], color='white')

    if highlight_line:
        hl = highlight_line_number + 10
        plt.plot([hl, hl], [0, 53.3], color='yellow')
        plt.text(hl + 2, 50, '<- {}'.format(highlighted_name),
                 color='yellow')
    return fig, ax

create_football_field()
plt.show()
import math
def get_dx_dy(angle, dist):
    cartesianAngleRadians = (450-angle)*math.pi/180.0
    dx = dist * math.cos(cartesianAngleRadians)
    dy = dist * math.sin(cartesianAngleRadians)
    return dx, dy

play_id = 20181007011551
fig, ax = create_football_field()
train_df.query("PlayId == @play_id and Team == 'away'") \
    .plot(x='X', y='Y', kind='scatter', ax=ax, color='orange', s=50, legend='Away')
train_df.query("PlayId == @play_id and Team == 'home'") \
    .plot(x='X', y='Y', kind='scatter', ax=ax, color='blue', s=50, legend='Home')
train_df.query("PlayId == @play_id and NflIdRusher == NflId") \
    .plot(x='X', y='Y', kind='scatter', ax=ax, color='red', s=100, legend='Rusher')
rusher_row = train_df.query("PlayId == @play_id and NflIdRusher == NflId")
yards_covered = rusher_row["Yards"].values[0]

x = rusher_row["X"].values[0]
y = rusher_row["Y"].values[0]
rusher_dir = rusher_row["Dir"].values[0]
rusher_speed = rusher_row["S"].values[0]
dx, dy = get_dx_dy(rusher_dir, rusher_speed)

ax.arrow(x, y, dx, dy, length_includes_head=True, width=0.3)
plt.title(f'Play # {play_id} and yard distance is {yards_covered}', fontsize=20)
plt.legend()
plt.show()
rusher_dir
def get_plot(play_id):
    fig, ax = create_football_field()
    train_df.query("PlayId == @play_id and Team == 'away'") \
        .plot(x='X', y='Y', kind='scatter', ax=ax, color='orange', s=50, legend='Away')
    train_df.query("PlayId == @play_id and Team == 'home'") \
        .plot(x='X', y='Y', kind='scatter', ax=ax, color='blue', s=50, legend='Home')
    train_df.query("PlayId == @play_id and NflIdRusher == NflId") \
        .plot(x='X', y='Y', kind='scatter', ax=ax, color='red', s=100, legend='Rusher')
    rusher_row = train_df.query("PlayId == @play_id and NflIdRusher == NflId")
    yards_covered = rusher_row["Yards"].values[0]

    x = rusher_row["X"].values[0]
    y = rusher_row["Y"].values[0]
    rusher_dir = rusher_row["Dir"].values[0]
    rusher_speed = rusher_row["S"].values[0]
    dx, dy = get_dx_dy(rusher_dir, rusher_speed)

    ax.arrow(x, y, dx, dy, length_includes_head=True, width=0.3)
    plt.title(f'Play # {play_id} and yard distance is {yards_covered}', fontsize=26)
    plt.legend()
    return plt

temp_df = train_df.groupby("PlayId").first()
temp_df = temp_df.sort_values(by="Yards").reset_index().head(5)

for play_id in temp_df["PlayId"].values:
    plt = get_plot(play_id)
    plt.show()
temp_df = train_df.groupby("PlayId").first()
temp_df = temp_df[temp_df["Yards"]==0].reset_index().head(5)

for play_id in temp_df["PlayId"].values:
    plt = get_plot(play_id)
    plt.show()
temp_df = train_df.groupby("PlayId").first()
temp_df = temp_df[temp_df["Yards"]>10].reset_index().head(5)

for play_id in temp_df["PlayId"].values:
    plt = get_plot(play_id)
    plt.show()
plt.figure(figsize=(16,12))
temp_df = train_df.query("NflIdRusher == NflId")
sns.scatterplot(temp_df["Dis"], temp_df["Yards"])
plt.xlabel('Distance covered', fontsize=12)
plt.ylabel('Yards (Target)', fontsize=12)
plt.title("Distance covered by Rusher Vs Yards (target)", fontsize=20)
plt.show()
plt.figure(figsize=(16,12))
temp_df = train_df.query("NflIdRusher == NflId")
sns.scatterplot(temp_df["S"], temp_df["Yards"])
plt.xlabel('Rusher Speed', fontsize=12)
plt.ylabel('Yards (Target)', fontsize=12)
plt.title("Rusher Speed Vs Yards (target)", fontsize=20)
plt.show()
plt.figure(figsize=(16,12))
temp_df = train_df.query("NflIdRusher == NflId")
sns.scatterplot(temp_df["A"], temp_df["Yards"])
plt.xlabel('Rusher Acceleration', fontsize=12)
plt.ylabel('Yards (Target)', fontsize=12)
plt.title("Rusher Acceleration Vs Yards (target)", fontsize=20)
plt.show()
plt.figure(figsize=(16,12))
temp_df = train_df.query("NflIdRusher == NflId")
sns.boxplot(data=temp_df, x="Position", y="Yards", showfliers=False, whis=3.0)
plt.xlabel('Rusher position', fontsize=12)
plt.ylabel('Yards (Target)', fontsize=12)
plt.title("Rusher Position Vs Yards (target)", fontsize=20)
plt.show()
plt.figure(figsize=(16,12))
temp_df = train_df.query("NflIdRusher == NflId")
sns.boxplot(data=temp_df, x="DefendersInTheBox", y="Yards", showfliers=False, whis=3.0)
plt.xlabel('Number of defenders in the box', fontsize=12)
plt.ylabel('Yards (Target)', fontsize=12)
plt.title("Count of defenders in the box Vs Yards (target)", fontsize=20)
plt.show()
plt.figure(figsize=(16,12))
temp_df = train_df.query("NflIdRusher == NflId")
sns.boxplot(data=temp_df, x="Down", y="Yards", showfliers=False)
plt.xlabel('Down Number', fontsize=12)
plt.ylabel('Yards (Target)', fontsize=12)
plt.title("Down Number Vs Yards (target)", fontsize=20)
plt.show()
plt.figure(figsize=(12,10))
temp_df = train_df.query("NflIdRusher == NflId")
sns.boxplot(data=temp_df, y="PossessionTeam", x="Yards", showfliers=False, whis=3.0)
plt.ylabel('PossessionTeam', fontsize=12)
plt.xlabel('Yards (Target)', fontsize=12)
plt.title("Possession team Vs Yards (target)", fontsize=20)
plt.show()
plt.figure(figsize=(16,12))
temp_df = train_df.query("NflIdRusher == NflId")
sns.catplot(data=temp_df, x="Quarter", y="Yards", kind="boxen")
plt.xlabel('Quarter', fontsize=12)
plt.ylabel('Yards (Target)', fontsize=12)
plt.title("Quarter Vs Yards (target)", fontsize=20)
plt.show()