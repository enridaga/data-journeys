
import pandas as pd   #importing all the important packages

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')
df =  pd.read_csv('../input/Pokemon.csv')  #read the csv file and save it into a variable

df.head(n=10)                    #print the first 10 rows of the table
df.columns = df.columns.str.upper().str.replace('_', '') #change into upper case

df.head()
df[df['LEGENDARY']==True].head(5)  #Showing the legendary pokemons
df = df.set_index('NAME') #change and set the index to the name attribute
## The index of Mega Pokemons contained extra and unneeded text. Removed all the text before "Mega"  

df.index = df.index.str.replace(".*(?=Mega)", "")

df.head(10)
df=df.drop(['#'],axis=1) #drop the columns with axis=1;axis=0 is for rows
print('The columns of the dataset are: ',df.columns) #show the dataframe columns

print('The shape of the dataframe is: ',df.shape)    #shape of the dataframe
#some values in TYPE2 are empty and thus they have to be filled or deleted

df['TYPE 2'].fillna(df['TYPE 1'], inplace=True) #fill NaN values in Type2 with corresponding values of Type
print(df.loc['Bulbasaur']) #retrieves complete row data from index with value Bulbasaur

print(df.iloc[0]) #retrieves complete row date from index 0 ; integer version of loc

print(df.ix[0]) #similar to iloc

print(df.ix['Kakuna']) #similar to loc

#loc works on labels in the index.

#iloc works on the positions in the index (so it only takes integers).

#ix usually tries to behave like loc but falls back to behaving like iloc if the label is not in the index.

#inoreder to find details about any pokemon, just specify its name
#filtering pokemons using logical operators

df[((df['TYPE 1']=='Fire') | (df['TYPE 1']=='Dragon')) & ((df['TYPE 2']=='Dragon') | (df['TYPE 2']=='Fire'))].head(3)
print("MAx HP:",df['HP'].argmax())  #returns the pokemon with highest HP

print("Max DEFENCE:",(df['DEFENSE']).idxmax()) #similar to argmax()
df.sort_values('TOTAL',ascending=False).head(3)  #this arranges the pokemons in the descendng order of the Totals.

#sort_values() is used for sorting and ascending=False is making it in descending order
print('The unique  pokemon types are',df['TYPE 1'].unique()) #shows all the unique types in column

print('The number of unique types are',df['TYPE 1'].nunique()) #shows count of unique values 
print(df['TYPE 1'].value_counts(), '\n' ,df['TYPE 2'].value_counts())#count different types of pokemons

df.groupby(['TYPE 1']).size()  #same as above

(df['TYPE 1']=='Bug').sum() #counts for a single value
df_summary = df.describe() #summary of the pokemon dataframe

df_summary
bins=range(0,200,20) #they act as containers

plt.hist(df["ATTACK"],bins,histtype="bar",rwidth=1.2,color='#0ff0ff') #hist() is used to plot a histogram

plt.xlabel('Attack') #set the xlabel name

plt.ylabel('Count') #set the ylabel name

plt.plot()

plt.axvline(df['ATTACK'].mean(),linestyle='dashed',color='red') #draw a vertical line showing the average Attack value

plt.show()
fire=df[(df['TYPE 1']=='Fire') | ((df['TYPE 2'])=="Fire")] #fire contains all fire pokemons

water=df[(df['TYPE 1']=='Water') | ((df['TYPE 2'])=="Water")]  #all water pokemins

plt.scatter(fire.ATTACK.head(50),fire.DEFENSE.head(50),color='R',label='Fire',marker="*",s=50) #scatter plot

plt.scatter(water.ATTACK.head(50),water.DEFENSE.head(50),color='B',label="Water",s=25)

plt.xlabel("Attack")

plt.ylabel("DEFENCE")

plt.legend()

plt.plot()

fig=plt.gcf()  #get the current figure using .gcf()

fig.set_size_inches(12,6) #set the size for the figure

plt.show()
strong=df.sort_values(by='TOTAL', ascending=False) #sorting the rows in descending order

strong.drop_duplicates(subset=['TYPE 1'],keep='first') #since the rows are now sorted in descending oredr

#thus we take the first row for every new type of pokemon i.e the table will check TYPE 1 of every pokemon

#The first pokemon of that type is the strongest for that type

#so we just keep the first row
labels = 'Water', 'Normal', 'Grass', 'Bug', 'Psychic', 'Fire', 'Electric', 'Rock', 'Other'

sizes = [112, 98, 70, 69, 57, 52, 44, 44, 175]

colors = ['Y', 'B', '#00ff00', 'C', 'R', 'G', 'silver', 'white', 'M']

explode = (0, 0, 0.1, 0, 0, 0, 0, 0, 0)  # only "explode" the 3rd slice 

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=90)

plt.axis('equal')

plt.title("Percentage of Different Types of Pokemon")

plt.plot()

fig=plt.gcf()

fig.set_size_inches(7,7)

plt.show()
df2=df.drop(['GENERATION','TOTAL'],axis=1)

sns.boxplot(data=df2)

plt.ylim(0,300)  #change the scale of y axix

plt.show()
plt.subplots(figsize = (15,5))

plt.title('Attack by Type1')

sns.boxplot(x = "TYPE 1", y = "ATTACK",data = df)

plt.ylim(0,200)

plt.show()
plt.subplots(figsize = (15,5))

plt.title('Attack by Type2')

sns.boxplot(x = "TYPE 2", y = "ATTACK",data=df)

plt.show()
plt.subplots(figsize = (15,5))

plt.title('Defence by Type')

sns.boxplot(x = "TYPE 1", y = "DEFENSE",data = df)

plt.show()
plt.subplots(figsize = (20,10))

plt.title('Attack by Type1')

sns.violinplot(x = "TYPE 1", y = "ATTACK",data = df)

plt.ylim(0,200)

plt.show()
plt.subplots(figsize = (20,10))

plt.title('Attack by Type1')

sns.violinplot(x = "TYPE 1", y = "DEFENSE",data = df)

plt.ylim(0,200)

plt.show()
plt.subplots(figsize = (15,5))

plt.title('Strongest Genaration')

sns.violinplot(x = "GENERATION", y = "TOTAL",data = df)

plt.show()
plt.figure(figsize=(12,6))

top_types=df['TYPE 1'].value_counts()[:10] #take the top 10 Types

df1=df[df['TYPE 1'].isin(top_types.index)] #take the pokemons of the type with highest numbers, top 10

sns.swarmplot(x='TYPE 1',y='TOTAL',data=df1,hue='LEGENDARY') # this plot shows the points belonging to individual pokemons

# It is distributed by Type

plt.axhline(df1['TOTAL'].mean(),color='red',linestyle='dashed')

plt.show()
plt.figure(figsize=(10,6)) #manage the size of the plot

sns.heatmap(df.corr(),annot=True) #df.corr() makes a correlation matrix and sns.heatmap is used to show the correlations heatmap

plt.show()
a=df.groupby(['GENERATION','TYPE 1']).count().reset_index()

a=a[['GENERATION','TYPE 1','TOTAL']]

a=a.pivot('GENERATION','TYPE 1','TOTAL')

a[['Water','Fire','Grass','Dragon','Normal','Rock','Flying','Electric']].plot(color=['b','r','g','#FFA500','brown','#6666ff','#001012','y'],marker='o')

fig=plt.gcf()

fig.set_size_inches(12,6)

plt.show()
a=df.groupby(['GENERATION','TYPE 2']).count().reset_index()

a=a[['GENERATION','TYPE 2','TOTAL']]

a=a.pivot('GENERATION','TYPE 2','TOTAL')

a[['Water','Fire','Grass','Dragon','Normal','Rock','Flying','Electric']].plot(color=['b','r','g','#FFA500','brown','#6666ff','#001012','y'],marker='o')

fig=plt.gcf()

fig.set_size_inches(12,6)

plt.show()