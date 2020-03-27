
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # visualize

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/student-mat.csv')
# I use final grade = G3, and weekly alcohol consumption = Dalc + Walc 

data.columns


plt.figure(figsize=(15,15))

sns.heatmap(data.corr(),annot = True,fmt = ".2f",cbar = True)

plt.xticks(rotation=90)

plt.yticks(rotation = 0)

data['Dalc'] = data['Dalc'] + data['Walc']
# There is no student who does not consume alcohol. However, all students at least 2 times in a week consume alcohol.

list = []

for i in range(11):

    list.append(len(data[data.Dalc == i]))

ax = sns.barplot(x = [0,1,2,3,4,5,6,7,8,9,10], y = list)

plt.ylabel('Number of Students')

plt.xlabel('Weekly alcohol consumption')
labels = ['2','3','4','5','6','7','8','9','10']

colors = ['lime','blue','orange','cyan','grey','purple','brown','red','darksalmon']

explode = [0,0,0,0,0,0,0,0,0]

sizes = []

for i in range(2,11):

    sizes.append(sum(data[data.Dalc == i].G3))

total_grade = sum(sizes)

average = total_grade/float(len(data))

plt.pie(sizes,explode=explode,colors=colors,labels=labels,autopct = '%1.1f%%')

plt.axis('equal')

plt.title('Total grade : '+str(total_grade))

plt.xlabel('Students grade distribution according to weekly alcohol consumption')

ave = sum(data.G3)/float(len(data))

data['ave_line'] = ave

data['average'] = ['above average' if i > ave else 'under average' for i in data.G3]

sns.swarmplot(x='Dalc', y = 'G3', hue = 'average',data= data,palette={'above average':'lime', 'under average': 'red'})
sum(data[data.Dalc == 2].G3)/float(len(data[data.Dalc == 2]))
# Average grade

list = []

for i in range(2,11):

    list.append(sum(data[data.Dalc == i].G3)/float(len(data[data.Dalc == i])))

ax = sns.barplot(x = [2,3,4,5,6,7,8,9,10], y = list)

plt.ylabel('Average Grades of students')

plt.xlabel('Weekly alcohol consumption')