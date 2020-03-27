
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import matplotlib.patches as patches

plt.style.use('seaborn')
color_pal = [x['color'] for x in plt.rcParams['axes.prop_cycle']]
pd.set_option('max_columns', 100) # So we can see more columns

# Read in the training data
train = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
# https://stackoverflow.com/questions/30228069/how-to-display-the-value-of-the-bar-on-each-bar-with-pyplot-barh
def label_bars(ax, bars, text_format, **kwargs):
    """
    Attaches a label on every bar of a regular or horizontal bar chart
    """
    ys = [bar.get_y() for bar in bars]
    y_is_constant = all(y == ys[0] for y in ys)  # -> regular bar chart, since all all bars start on the same y level (0)

    if y_is_constant:
        _label_bar(ax, bars, text_format, **kwargs)
    else:
        _label_barh(ax, bars, text_format, **kwargs)


def _label_bar(ax, bars, text_format, **kwargs):
    """
    Attach a text label to each bar displaying its y value
    """
    max_y_value = ax.get_ylim()[1]
    inside_distance = max_y_value * 0.05
    outside_distance = max_y_value * 0.01

    for bar in bars:
        text = text_format.format(bar.get_height())
        text_x = bar.get_x() + bar.get_width() / 2

        is_inside = bar.get_height() >= max_y_value * 0.15
        if is_inside:
            color = "white"
            text_y = bar.get_height() - inside_distance
        else:
            color = "black"
            text_y = bar.get_height() + outside_distance

        ax.text(text_x, text_y, text, ha='center', va='bottom', color=color, **kwargs)


def _label_barh(ax, bars, text_format, **kwargs):
    """
    Attach a text label to each bar displaying its y value
    Note: label always outside. otherwise it's too hard to control as numbers can be very long
    """
    max_x_value = ax.get_xlim()[1]
    distance = max_x_value * 0.0025

    for bar in bars:
        text = text_format.format(bar.get_width())

        text_x = bar.get_width() + distance
        text_y = bar.get_y() + bar.get_height() / 2

        ax.text(text_x, text_y, text, va='center', **kwargs)
train.groupby('PlayId').first()['Yards'] \
    .plot(kind='hist',
          figsize=(15, 5),
          bins=50,
          title='Distribution of Yards Gained (Target)')
plt.show()
fig, axes = plt.subplots(4, 1, figsize=(15, 8), sharex=True)
n = 0
for i, d in train.groupby('Down'):
    d['Yards'].plot(kind='hist',
                    bins=30,
                   color=color_pal[n],
                   ax=axes[n],
                   title=f'Yards Gained on down {i}')
    n+=1
fig, ax = plt.subplots(figsize=(20, 5))
sns.violinplot(x='Distance-to-Gain',
               y='Yards',
               data=train.rename(columns={'Distance':'Distance-to-Gain'}),
               ax=ax)
plt.ylim(-10, 20)
plt.title('Yards vs Distance-to-Gain')
plt.show()
print('Unique game data provided: {}'.format(train['GameId'].nunique()))
print('Unique play data provided: {}'.format(train['PlayId'].nunique()))
train.groupby('GameId')['PlayId'] \
    .nunique() \
    .plot(kind='hist', figsize=(15, 5),
         title='Distribution of Plays per GameId',
         bins=50)
plt.show()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
sns.boxplot(data=train.groupby('PlayId').first()[['Distance','Down']],
            x='Down', y='Distance', ax=ax1)
ax1.set_title('Distance-to-Gain by Down')
sns.boxplot(data=train.groupby('PlayId').first()[['Yards','Down']],
            x='Down', y='Yards', ax=ax2)
ax2.set_title('Yards Gained by Down')
plt.show()
train['Distance'].plot(kind='hist',
                       title='Distribution of Distance to Go',
                       figsize=(15, 5),
                       bins=30,
                       color=color_pal[2])
plt.show()
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
train['S'].plot(kind='hist', ax=ax1,
                title='Distribution of Speed',
                bins=20,
                color=color_pal[0])
train['A'].plot(kind='hist',
                ax=ax2,
                title='Distribution of Acceleration',
                bins=20,
                color=color_pal[1])
train['Dis'].plot(kind='hist',
                  ax=ax3,
                  title='Distribution of Distance',
                  bins=20,
                  color=color_pal[2])
plt.show()
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
train.query("NflIdRusher == NflId")['S'] \
    .plot(kind='hist',
          ax=ax1,
          title='Distribution of Speed (Ball Carrier Only)',
          bins=20,
          color=color_pal[0])
train.query("NflIdRusher == NflId")['A'] \
    .plot(kind='hist',
          ax=ax2,
          title='Distribution of Acceleration (Ball Carrier Only)',
          bins=20,
          color=color_pal[1])
train.query("NflIdRusher == NflId")['Dis'] \
    .plot(kind='hist',
          ax=ax3,
          title='Distribution of Distance (Ball Carrier Only)',
          bins=20,
          color=color_pal[2])
plt.show()
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

sns.pairplot(train.query("NflIdRusher == NflId").sample(1000)[['S','Dis','A','Yards','DefensePersonnel']],
            hue='DefensePersonnel')
plt.suptitle('Speed, Acceleration, Distance traveled, and Yards Gained for Rushers')
plt.show()
fig, ax = plt.subplots(1, 2, figsize=(20, 8))
train.groupby('PlayId') \
    .first() \
    .groupby('OffensePersonnel') \
    .count()['GameId'] \
    .sort_values(ascending=False) \
    .head(15) \
    .sort_values() \
    .plot(kind='barh',
         title='Offense Personnel # of Plays',
         ax=ax[0])
train.groupby('PlayId') \
    .first() \
    .groupby('DefensePersonnel') \
    .count()['GameId'] \
    .sort_values(ascending=False) \
    .head(15) \
    .sort_values() \
    .plot(kind='barh',
         title='Defense Personnel # of Plays',
         ax=ax[1])
plt.show()
top_10_defenses = train.groupby('DefensePersonnel')['GameId'] \
    .count() \
    .sort_values(ascending=False).index[:10] \
    .tolist()
train_play = train.groupby('PlayId').first()
train_top10_def = train_play.loc[train_play['DefensePersonnel'].isin(top_10_defenses)]

fig, ax = plt.subplots(figsize=(15, 5))
sns.violinplot(x='DefensePersonnel',
               y='Yards',
               data=train_top10_def,
               ax=ax)
plt.ylim(-10, 20)
plt.title('Yards vs Defensive Personnel')
plt.show()
fig, ax = plt.subplots(figsize=(15, 5))
ax.set_ylim(-10, 60)
ax.set_title('Yards vs Quarter')
sns.boxenplot(x='Quarter',
            y='Yards',
            data=train.sample(5000),
            ax=ax)
plt.show()
fig, ax = plt.subplots(figsize=(15, 5))
ax.set_ylim(-10, 60)
sns.boxenplot(x='DefendersInTheBox',
               y='Yards',
               data=train.query('DefendersInTheBox > 2'),
               ax=ax)
plt.title('Yards vs Defenders in the Box')
plt.show()
fig, axes = plt.subplots(3, 2, constrained_layout=True, figsize=(15 , 10))
#fig.tight_layout()
ax_idx = 0
ax_idx2 = 0
for i in range(4, 10):
    this_ax = axes[ax_idx2][ax_idx]
    #print(ax_idx, ax_idx2)
    sns.distplot(train.query('DefendersInTheBox == @i')['Yards'],
                ax=this_ax,
                color=color_pal[ax_idx2])
    this_ax.set_title(f'{i} Defenders in the box')
    this_ax.set_xlim(-10, 20)
    ax_idx += 1
    if ax_idx == 2:
        ax_idx = 0
        ax_idx2 += 1
plt.show()
train.query("NflIdRusher == NflId") \
    .groupby('DisplayName')['Yards'] \
    .agg(['count','mean']) \
    .query('count > 100') \
    .sort_values('mean', ascending=True) \
    .tail(10)['mean'] \
    .plot(kind='barh',
          figsize=(15, 5),
          color=color_pal[5],
          xlim=(0,6),
          title='Top 10 Players by Average Yards')
plt.show()
train.query("NflIdRusher == NflId") \
    .groupby('DisplayName')['Yards'] \
    .agg(['count','mean']) \
    .query('count > 100') \
    .sort_values('mean', ascending=True) \
    .head(10)['mean'] \
    .plot(kind='barh',
          figsize=(15, 5),
          color=color_pal[0],
          xlim=(0,6),
          title='Bottom 10 Players by Average Yards')
plt.show()
# Create the DL-LB combos
train['DL_LB'] = train['DefensePersonnel'] \
    .str[:10] \
    .str.replace(' DL, ','-') \
    .str.replace(' LB','') # Clean up and convert to DL-LB combo
top_5_dl_lb_combos = train.groupby('DL_LB').count()['GameId'] \
    .sort_values() \
    .tail(10).index.tolist()
ax = train.loc[train['DL_LB'].isin(top_5_dl_lb_combos)] \
    .groupby('DL_LB').mean()['Yards'] \
    .sort_values(ascending=True) \
    .plot(kind='bar',
          title='Average Yards Top 5 Defensive DL-LB combos',
          figsize=(15, 5),
          color=color_pal[4])
# for p in ax.patches:
#     ax.annotate(str(round(p.get_height(), 2)),
#                 (p.get_x() * 1.005, p.get_height() * 1.015))

#bars = ax.bar(0.5, 5, width=0.5, align="center")
bars = [p for p in ax.patches]
value_format = "{:0.2f}"
label_bars(ax, bars, value_format, fontweight='bold')
plt.show()
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

import math
def get_dx_dy(angle, dist):
    cartesianAngleRadians = (450-angle)*math.pi/180.0
    dx = dist * math.cos(cartesianAngleRadians)
    dy = dist * math.sin(cartesianAngleRadians)
    return dx, dy
play_id = train.query("DL_LB == '3-4'")['PlayId'].reset_index(drop=True)[500]
fig, ax = create_football_field()
train.query("PlayId == @play_id and Team == 'away'") \
    .plot(x='X', y='Y', kind='scatter', ax=ax, color='orange', s=200, legend='Away')
train.query("PlayId == @play_id and Team == 'home'") \
    .plot(x='X', y='Y', kind='scatter', ax=ax, color='blue', s=200, legend='Home')
train.query("PlayId == @play_id and NflIdRusher == NflId") \
    .plot(x='X', y='Y', kind='scatter', ax=ax, color='red', s=200, legend='Rusher')
rusher_row = train.query("PlayId == @play_id and NflIdRusher == NflId")
yards_covered = rusher_row["Yards"].values[0]

x = rusher_row["X"].values[0]
y = rusher_row["Y"].values[0]
rusher_dir = rusher_row["Dir"].values[0]
rusher_speed = rusher_row["S"].values[0]
dx, dy = get_dx_dy(rusher_dir, rusher_speed)
yards_gained = train.query("PlayId == @play_id")['Yards'].tolist()[0]
ax.arrow(x, y, dx, dy, length_includes_head=True, width=0.3)
plt.title(f'Example of a 3-4 Defense - run resulted in {yards_gained} yards gained', fontsize=20)
plt.legend()
plt.show()
play_id = train.query("DL_LB == '4-3'")['PlayId'].reset_index(drop=True)[500]
fig, ax = create_football_field()
train.query("PlayId == @play_id and Team == 'away'") \
    .plot(x='X', y='Y', kind='scatter', ax=ax, color='orange', s=200, legend='Away')
train.query("PlayId == @play_id and Team == 'home'") \
    .plot(x='X', y='Y', kind='scatter', ax=ax, color='blue', s=200, legend='Home')
train.query("PlayId == @play_id and NflIdRusher == NflId") \
    .plot(x='X', y='Y', kind='scatter', ax=ax, color='red', s=200, legend='Rusher')
rusher_row = train.query("PlayId == @play_id and NflIdRusher == NflId")
yards_covered = rusher_row["Yards"].values[0]

x = rusher_row["X"].values[0]
y = rusher_row["Y"].values[0]
rusher_dir = rusher_row["Dir"].values[0]
rusher_speed = rusher_row["S"].values[0]
dx, dy = get_dx_dy(rusher_dir, rusher_speed)
yards_gained = train.query("PlayId == @play_id")['Yards'].tolist()[0]
ax.arrow(x, y, dx, dy, length_includes_head=True, width=0.3)
plt.title(f'Example of a 4-3 Defense - run resulted in {yards_gained} yard gained', fontsize=20)
plt.legend()
plt.show()
play_id = train.query("DL_LB == '4-2'")['PlayId'].reset_index(drop=True)[500]
fig, ax = create_football_field()
train.query("PlayId == @play_id and Team == 'away'") \
    .plot(x='X', y='Y', kind='scatter', ax=ax, color='orange', s=200, legend='Away')
train.query("PlayId == @play_id and Team == 'home'") \
    .plot(x='X', y='Y', kind='scatter', ax=ax, color='blue', s=200, legend='Home')
train.query("PlayId == @play_id and NflIdRusher == NflId") \
    .plot(x='X', y='Y', kind='scatter', ax=ax, color='red', s=200, legend='Rusher')
rusher_row = train.query("PlayId == @play_id and NflIdRusher == NflId")
yards_covered = rusher_row["Yards"].values[0]

x = rusher_row["X"].values[0]
y = rusher_row["Y"].values[0]
rusher_dir = rusher_row["Dir"].values[0]
rusher_speed = rusher_row["S"].values[0]
dx, dy = get_dx_dy(rusher_dir, rusher_speed)
yards_gained = train.query("PlayId == @play_id")['Yards'].tolist()[0]
ax.arrow(x, y, dx, dy, length_includes_head=True, width=0.3)
plt.title(f'Example of a 4-2 Defense - run resulted in {yards_gained} yards gained', fontsize=20)
plt.legend()
plt.show()
train['SnapHandoffSeconds'] = (pd.to_datetime(train['TimeHandoff']) - \
                               pd.to_datetime(train['TimeSnap'])).dt.total_seconds()

(train.groupby('SnapHandoffSeconds').count() / 22 )['GameId'].plot(kind='bar',
                                                                   figsize=(15, 5))
bars = [p for p in ax.patches]
value_format = "{}"
label_bars(ax, bars, value_format, fontweight='bold')
plt.show()
train.groupby('SnapHandoffSeconds')['Yards'].mean().plot(kind='barh',
                                                         color=color_pal[1],
                                                         figsize=(15, 5),
                                                         title='Average Yards Gained by SnapHandoff Seconds')
plt.show()
