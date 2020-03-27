
import pandas as pd
import numpy as np
pd.options.display.max_columns = 100
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import seaborn as sns
import datetime, tqdm
from kaggle.competitions import nflrush

import math

from scipy.spatial import Voronoi, voronoi_plot_2d
train = pd.read_csv("../input/nfl-big-data-bowl-2020/train.csv", low_memory=False)
test = pd.read_csv('../input/dsbowl2019-test/test.csv', low_memory=False)
set(train.columns) - set(test.columns)
test['Yards'] = np.NaN
train = pd.concat((train, test[train.columns]), axis=0)
train['ToLeft'] = train.PlayDirection == "left"
train['IsBallCarrier'] = train.NflId == train.NflIdRusher

train.loc[train.VisitorTeamAbbr == "ARI", 'VisitorTeamAbbr'] = "ARZ"
train.loc[train.HomeTeamAbbr == "ARI", 'HomeTeamAbbr'] = "ARZ"

train.loc[train.VisitorTeamAbbr == "BAL", 'VisitorTeamAbbr'] = "BLT"
train.loc[train.HomeTeamAbbr == "BAL", 'HomeTeamAbbr'] = "BLT"

train.loc[train.VisitorTeamAbbr == "CLE", 'VisitorTeamAbbr'] = "CLV"
train.loc[train.HomeTeamAbbr == "CLE", 'HomeTeamAbbr'] = "CLV"

train.loc[train.VisitorTeamAbbr == "HOU", 'VisitorTeamAbbr'] = "HST"
train.loc[train.HomeTeamAbbr == "HOU", 'HomeTeamAbbr'] = "HST"

train['Dir_rad'] = np.mod(90 - train.Dir, 360) * math.pi/180.0
    
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
                             edgecolor='r', facecolor='darkgreen', zorder=0,  alpha=0.5)

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
def get_dx_dy(radian_angle, dist):
    dx = dist * math.cos(radian_angle)
    dy = dist * math.sin(radian_angle)
    return dx, dy

def show_play(play_id, train=train):
    df = train[train.PlayId == play_id]
    fig, ax = create_football_field()
    ax.scatter(df.X, df.Y, cmap='rainbow', c=~(df.Team == 'home'), s=100)
    rusher_row = df[df.NflIdRusher == df.NflId]
    ax.scatter(rusher_row.X, rusher_row.Y, color='black')
    yards_covered = rusher_row["Yards"].values[0]
    x = rusher_row["X"].values[0]
    y = rusher_row["Y"].values[0]
    rusher_dir = rusher_row["Dir_rad"].values[0]
    rusher_speed = rusher_row["S"].values[0]
    dx, dy = get_dx_dy(rusher_dir, rusher_speed)

    ax.arrow(x, y, dx, dy, length_includes_head=True, width=0.3, color='black')
    left = 'left' if df.ToLeft.sum() > 0 else 'right'
    plt.title(f'Play # {play_id} moving to {left}, yard distance is {yards_covered}', fontsize=20)
    plt.legend()
    plt.show()
show_play(20170910001102)
show_play(20170910000081)
train['TeamOnOffense'] = "home"
train.loc[train.PossessionTeam != train.HomeTeamAbbr, 'TeamOnOffense'] = "away"
train['IsOnOffense'] = train.Team == train.TeamOnOffense # Is player on offense?
train['YardLine_std'] = 100 - train.YardLine
train.loc[train.FieldPosition.fillna('') == train.PossessionTeam,  
          'YardLine_std'
         ] = train.loc[train.FieldPosition.fillna('') == train.PossessionTeam,  
          'YardLine']
train['X_std'] = train.X
train.loc[train.ToLeft, 'X_std'] = 120 - train.loc[train.ToLeft, 'X'] 
train['Y_std'] = train.Y
train.loc[train.ToLeft, 'Y_std'] = 160/3 - train.loc[train.ToLeft, 'Y'] 
#train['Orientation_std'] = -90 + train.Orientation
#train.loc[train.ToLeft, 'Orientation_std'] = np.mod(180 + train.loc[train.ToLeft, 'Orientation_std'], 360)
train['Dir_std'] = train.Dir_rad
train.loc[train.ToLeft, 'Dir_std'] = np.mod(np.pi + train.loc[train.ToLeft, 'Dir_rad'], 2*np.pi)

def show_play_std(play_id, train=train):
    df = train[train.PlayId == play_id]
    YardLine = df.YardLine_std.values[0]
    fig, ax = create_football_field(highlight_line=True,
                                    highlight_line_number=YardLine,
                                   )
    ax.scatter(df.X_std, df.Y_std, cmap='rainbow', c=~df.IsOnOffense, s=100)
    rusher_row = df[df.NflIdRusher == df.NflId]
    ax.scatter(rusher_row.X_std, rusher_row.Y_std, color='black', s=100)
    yards_covered = rusher_row["Yards"].values[0]

    for (x, y, Dir, S) in zip(df.X_std, df.Y_std, df.Dir_std, df.S):       
        dx, dy = get_dx_dy(Dir, S)
        ax.arrow(x, y, dx, dy, length_includes_head=False, width=0.2, color='black', alpha=0.5)
    left = 'left' if df.ToLeft.sum() > 0 else 'right'
    plt.title(f'Play # {play_id} moving to {left}, yard distance is {yards_covered}, scrimage is {YardLine} ', fontsize=20)
    plt.legend()
    plt.show()
for play_id in [20170910001102, 20170910000081]: 
    show_play_std(play_id)
plt.figure(figsize=(12, 12))
ax = plt.subplot(221, projection='polar')
df = train[train.ToLeft & train.IsBallCarrier]
ax.set_title('Carriers direction with team moving to left')
ax.scatter(df.Dir_rad, df.S, s=1, alpha=0.2)

ax = plt.subplot(222, projection='polar')
df = train[~train.ToLeft & train.IsBallCarrier]
ax.set_title('Carriers direction with team moving to right')
ax.scatter(df.Dir_rad, df.S, s=1, alpha=0.2)

ax = plt.subplot(223, projection='polar')
df = train[train.IsOnOffense & train.IsBallCarrier]
ax.set_title('Carriers direction')
ax.scatter(df.Dir_rad, df.S, s=1, alpha=0.2)

ax = plt.subplot(224, projection='polar')
df = train[train.IsOnOffense & train.IsBallCarrier]
ax.set_title('Carriers standardised direction')
ax.scatter(df.Dir_std, df.S, s=1, alpha=0.2)

train['Orientation_rad'] = np.mod(train.Orientation, 360) * math.pi/180.0

plt.figure(figsize=(12, 18))
ax = plt.subplot(321, projection='polar')
df = train[train.ToLeft & train.IsBallCarrier & (train.Season == 2017)]
ax.set_title('Carriers orientation with team moving to left in 2017')
ax.scatter(df.Orientation_rad, df.S, s=1, alpha=0.2)

ax = plt.subplot(322, projection='polar')
df = train[~train.ToLeft & train.IsBallCarrier & (train.Season == 2017)]
ax.set_title('Carriers orientation with team moving to right in 2017')
ax.scatter(df.Orientation_rad, df.S, s=1, alpha=0.2)

ax = plt.subplot(323, projection='polar')
df = train[train.ToLeft & train.IsBallCarrier & (train.Season == 2018)]
ax.set_title('Carriers orientation with team moving to left in 2018')
ax.scatter(df.Orientation_rad, df.S, s=1, alpha=0.2)

ax = plt.subplot(324, projection='polar')
df = train[~train.ToLeft & train.IsBallCarrier & (train.Season == 2018)]
ax.set_title('Carriers orientation with team moving to right in 2018')
ax.scatter(df.Orientation_rad, df.S, s=1, alpha=0.2)


ax = plt.subplot(325, projection='polar')
df = train[train.ToLeft & train.IsBallCarrier & (train.Season == 2019)]
ax.set_title('Carriers orientation with team moving to left in 2019')
ax.scatter(df.Orientation_rad, df.S, s=1, alpha=0.2)

ax = plt.subplot(326, projection='polar')
df = train[~train.ToLeft & train.IsBallCarrier & (train.Season == 2019)]
ax.set_title('Carriers orientation with team moving to right in 2019')
ax.scatter(df.Orientation_rad, df.S, s=1, alpha=0.2)

train.loc[train.Season >= 2018, 'Orientation_rad'
         ] = np.mod(train.loc[train.Season >= 2018, 'Orientation'] - 90, 360) * math.pi/180.0

plt.figure(figsize=(12, 18))
ax = plt.subplot(321, projection='polar')
df = train[train.ToLeft & train.IsBallCarrier & (train.Season == 2017)]
ax.set_title('Carriers orientation with team moving to left in 2017')
ax.scatter(df.Orientation_rad, df.S, s=1, alpha=0.2)

ax = plt.subplot(322, projection='polar')
df = train[~train.ToLeft & train.IsBallCarrier & (train.Season == 2017)]
ax.set_title('Carriers orientation with team moving to right in 2017')
ax.scatter(df.Orientation_rad, df.S, s=1, alpha=0.2)

ax = plt.subplot(323, projection='polar')
df = train[train.ToLeft & train.IsBallCarrier & (train.Season == 2018)]
ax.set_title('Carriers orientation with team moving to left in 2018')
ax.scatter(df.Orientation_rad, df.S, s=1, alpha=0.2)

ax = plt.subplot(324, projection='polar')
df = train[~train.ToLeft & train.IsBallCarrier & (train.Season == 2018)]
ax.set_title('Carriers orientation with team moving to right in 2018')
ax.scatter(df.Orientation_rad, df.S, s=1, alpha=0.2)

ax = plt.subplot(325, projection='polar')
df = train[train.ToLeft & train.IsBallCarrier & (train.Season == 2019)]
ax.set_title('Carriers orientation with team moving to left in 2019')
ax.scatter(df.Orientation_rad, df.S, s=1, alpha=0.2)

ax = plt.subplot(326, projection='polar')
df = train[~train.ToLeft & train.IsBallCarrier & (train.Season == 2019)]
ax.set_title('Carriers orientation with team moving to right in 2019')
ax.scatter(df.Orientation_rad, df.S, s=1, alpha=0.2)

train['Orientation_rad'] = np.mod(train.Orientation, 360) * math.pi/180.0
train.loc[train.Season >= 2018, 'Orientation_rad'
         ] = np.mod(train.loc[train.Season >= 2018, 'Orientation'] - 90, 360) * math.pi/180.0
train['Orientation_std'] = train.Orientation_rad
train.loc[train.ToLeft, 'Orientation_std'] = np.mod(np.pi + train.loc[train.ToLeft, 'Orientation_rad'], 2*np.pi)

plt.figure(figsize=(12, 12))
ax = plt.subplot(221, projection='polar')
df = train[train.ToLeft & train.IsBallCarrier]
ax.set_title('Carriers orientation with team moving to left')
ax.scatter(df.Orientation_rad, df.S, s=1, alpha=0.2)

ax = plt.subplot(222, projection='polar')
df = train[~train.ToLeft & train.IsBallCarrier]
ax.set_title('Carriers orientation with team moving to right')
ax.scatter(df.Orientation_rad, df.S, s=1, alpha=0.2)

ax = plt.subplot(223, projection='polar')
df = train[train.IsOnOffense & train.IsBallCarrier]
ax.set_title('Carriers orientation')
ax.scatter(df.Orientation_rad, df.S, s=1, alpha=0.2)

ax = plt.subplot(224, projection='polar')
df = train[train.IsOnOffense & train.IsBallCarrier]
ax.set_title('Carriers standardised orientation')
ax.scatter(df.Orientation_std, df.S, s=1, alpha=0.2)
def show_voronoi(play_id, train=train):
    df = train[train.PlayId == play_id]
    xy = df[['X_std', 'Y_std']].values
    n_points = xy.shape[0]
    offense = df.IsOnOffense.values
    vor = Voronoi(xy)
    fig, ax = plt.subplots(1)
    ax.axis('equal')
    voronoi_plot_2d(vor, ax=ax, show_points=False, show_vertices=False)
    for r in range(n_points):
        region = vor.regions[vor.point_region[r]]
        if not -1 in region:
            polygon = [vor.vertices[i] for i in region]
            if offense[r]:
                plt.fill(*zip(*polygon), c='b', alpha=0.25)
            else:
                plt.fill(*zip(*polygon), c='r', alpha=0.25)
    ax.scatter(df.X_std, df.Y_std, cmap='rainbow', c=~df.IsOnOffense, s=10)
    rusher_row = df[df.NflIdRusher == df.NflId]
    ax.scatter(rusher_row.X_std, rusher_row.Y_std, color='black', s=100)
for play_id in [20170910000081, 20170910001102, ]: 
    vor = show_voronoi(play_id)
def show_voronoi(play_id, train=train):
    df = train[train.PlayId == play_id]
    xy = df[['X_std', 'Y_std']].values
    n_points = xy.shape[0]
    xy1 = xy.copy()
    xy1[:,1] = - xy[:,1]
    xy2 = xy.copy()
    xy2[:,1] = 320/3 - xy[:,1]
    xy3 = xy.copy()
    xy3[:,0] = 20 - xy[:,0]
    xy4 = xy.copy()
    xy4[:,0] = 220 - xy[:,0]
    xy = np.concatenate((xy, xy1, xy2, xy3, xy4), axis=0)
    offense = df.IsOnOffense.values
    vor = Voronoi(xy)
    fig, ax = plt.subplots(1)
    ax.axis('equal')
    #voronoi_plot_2d(vor, ax=ax, show_points=False, show_vertices=False)
    for r in range(n_points):
        region = vor.regions[vor.point_region[r]]
        if not -1 in region:
            polygon = [vor.vertices[i] for i in region]
            if offense[r]:
                plt.fill(*zip(*polygon), c='b', alpha=0.25)
            else:
                plt.fill(*zip(*polygon), c='r', alpha=0.25)
    ax.scatter(df.X_std, df.Y_std, cmap='rainbow', c=~df.IsOnOffense, s=10)
    rusher_row = df[df.NflIdRusher == df.NflId]
    ax.scatter(rusher_row.X_std, rusher_row.Y_std, color='black', s=100)
for play_id in [20170910000081, 20170910001102, ]: 
    vor = show_voronoi(play_id)
def show_voronoi_rusher(play_id, train=train):
    df = train[train.PlayId == play_id]
    df = df[(df.NflIdRusher == df.NflId) | ~df.IsOnOffense]
    xy = df[['X_std', 'Y_std']].values
    n_points = xy.shape[0]
    xy1 = xy.copy()
    xy1[:,1] = - xy[:,1]
    xy2 = xy.copy()
    xy2[:,1] = 320/3 - xy[:,1]
    xy3 = xy.copy()
    xy3[:,0] = 20 - xy[:,0]
    xy4 = xy.copy()
    xy4[:,0] = 220 - xy[:,0]
    xy = np.concatenate((xy, xy1, xy2, xy3, xy4), axis=0)
    offense = df.IsOnOffense.values
    vor = Voronoi(xy)
    fig, ax = plt.subplots(1)
    ax.axis('equal')
    for r in range(n_points):
        region = vor.regions[vor.point_region[r]]
        if not -1 in region:
            polygon = [vor.vertices[i] for i in region]
            if offense[r]:
                plt.fill(*zip(*polygon), c='b', alpha=0.25)
            else:
                plt.fill(*zip(*polygon), c='r', alpha=0.25)
    ax.scatter(df.X_std, df.Y_std, cmap='rainbow', c=~df.IsOnOffense, s=10)
    rusher_row = df[df.NflIdRusher == df.NflId]
    ax.scatter(rusher_row.X_std, rusher_row.Y_std, color='black', s=100)

for play_id in [20170910000081, 20170910001102, ]: 
    vor = show_voronoi_rusher(play_id)
