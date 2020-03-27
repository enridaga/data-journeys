
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
pkmn = pd.read_csv('../input/Pokemon.csv')
pkmn.head()
pkmn = pkmn.drop(['Generation', 'Legendary'],1)
sns.jointplot(x="HP", y="Attack", data=pkmn);
sns.boxplot(y="HP", data=pkmn);
sns.boxplot(data=pkmn);
pkmn = pkmn.drop(['Total', '#'],1)
sns.boxplot(data=pkmn);
pkmn = pd.melt(pkmn, id_vars=["Name", "Type 1", "Type 2"], var_name="Stat")
pkmn.head()
sns.swarmplot(x="Stat", y="value", data=pkmn, hue="Type 1");
plt.figure(figsize=(12,10))

plt.ylim(0, 275)

sns.swarmplot(x="Stat", y="value", data=pkmn, hue="Type 1", split=True, size=7)

plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.);
sns.set_style("whitegrid")

with sns.color_palette([

    "#8ED752", "#F95643", "#53AFFE", "#C3D221", "#BBBDAF",

    "#AD5CA2", "#F8E64E", "#F0CA42", "#F9AEFE", "#A35449",

    "#FB61B4", "#CDBD72", "#7673DA", "#66EBFF", "#8B76FF",

    "#8E6856", "#C3C1D7", "#75A4F9"], n_colors=18, desat=.9):

    plt.figure(figsize=(12,10))

    plt.ylim(0, 275)

    sns.swarmplot(x="Stat", y="value", data=pkmn, hue="Type 1", split=True, size=7)

    plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.);