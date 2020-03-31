
from IPython.display import YouTubeVideo
YouTubeVideo('QOO14BQITz4', width=700, height=400)
# IMPORTING LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
from PIL import  Image
### matplotlib inline
import pandas as pd
import seaborn as sns
import itertools
import warnings
warnings.filterwarnings("ignore")
from wordcloud import WordCloud,STOPWORDS
import io
import base64
from matplotlib import rc,animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.basemap import Basemap
import folium
import folium.plugins
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import os
print(os.listdir("../input"))
application_train     = pd.read_csv(r"../input/home-credit-default-risk/application_train.csv")
application_test      = pd.read_csv(r"../input/home-credit-default-risk/application_test.csv")
bureau                = pd.read_csv(r"../input/home-credit-default-risk/bureau.csv")
bureau_balance        = pd.read_csv(r"../input/home-credit-default-risk/bureau_balance.csv")
credit_card_balance   = pd.read_csv(r"../input/home-credit-default-risk/credit_card_balance.csv")
installments_payments = pd.read_csv(r"../input/home-credit-default-risk/installments_payments.csv")
pos_cash_balance      = pd.read_csv(r"../input/home-credit-default-risk/POS_CASH_balance.csv")
previous_application  = pd.read_csv(r"../input/home-credit-default-risk/previous_application.csv")
print ("application_train     :",application_train.shape)
print ("application_test      :",application_test.shape)
print ("bureau                :",bureau.shape)
print ("bureau_balance        :",bureau_balance.shape)
print ("credit_card_balance   :",credit_card_balance.shape)
print ("installments_payments :",installments_payments.shape)
print ("pos_cash_balance      :",pos_cash_balance.shape)
print ("previous_application  :",previous_application.shape)
display("application_train")
display(application_train.head(3))
display("application_test")
display(application_test.head(3))
display("bureau")
display(bureau.head(3))
display("bureau_balance")
display(bureau_balance.head(3))
display("credit_card_balance")
display(credit_card_balance.head(3))
display("installments_payments")
display(installments_payments.head(3))
display("pos_cash_balance")
display(pos_cash_balance.head(3))
display("previous_application")
display(previous_application.head(3))
image = np.array(Image.open(r"../input/picture-credit/home_credit.png"))
fig = plt.figure(figsize=(15,8))
plt.imshow(image,interpolation="bilinear")
plt.axis("off")
fig.set_facecolor("lightgrey")
plt.title("Data Sequence")
plt.show()
fig = plt.figure(figsize=(18,6))
miss_train = pd.DataFrame((application_train.isnull().sum())*100/application_train.shape[0]).reset_index()
miss_test = pd.DataFrame((application_test.isnull().sum())*100/application_test.shape[0]).reset_index()
miss_train["type"] = "train"
miss_test["type"]  =  "test"
missing = pd.concat([miss_train,miss_test],axis=0)
ax = sns.pointplot("index",0,data=missing,hue="type")
plt.xticks(rotation =90,fontsize =7)
plt.title("Percentage of Missing values in application train and test data")
plt.ylabel("PERCENTAGE")
plt.xlabel("COLUMNS")
ax.set_facecolor("k")
fig.set_facecolor("lightgrey")
plt.figure(figsize=(15,20))

plt.subplot(231)
sns.heatmap(pd.DataFrame(bureau.isnull().sum()/bureau.shape[0]*100),annot=True,
            cmap=sns.color_palette("cool"),linewidth=1,linecolor="white")
plt.title("bureau")

plt.subplot(232)
sns.heatmap(pd.DataFrame(bureau_balance.isnull().sum()/bureau_balance.shape[0]*100),annot=True,
            cmap=sns.color_palette("cool"),linewidth=1,linecolor="white")
plt.title("bureau_balance")

plt.subplot(233)
sns.heatmap(pd.DataFrame(credit_card_balance.isnull().sum()/credit_card_balance.shape[0]*100),annot=True,
            cmap=sns.color_palette("cool"),linewidth=1,linecolor="white")
plt.title("credit_card_balance")

plt.subplot(234)
sns.heatmap(pd.DataFrame(installments_payments.isnull().sum()/installments_payments.shape[0]*100),annot=True,
            cmap=sns.color_palette("cool"),linewidth=1,linecolor="white")
plt.title("installments_payments")

plt.subplot(235)
sns.heatmap(pd.DataFrame(pos_cash_balance.isnull().sum()/pos_cash_balance.shape[0]*100),annot=True,
            cmap=sns.color_palette("cool"),linewidth=1,linecolor="white")
plt.title("pos_cash_balance")

plt.subplot(236)
sns.heatmap(pd.DataFrame(previous_application.isnull().sum()/previous_application.shape[0]*100),annot=True,
            cmap=sns.color_palette("cool"),linewidth=1,linecolor="white")
plt.title("previous_application")

plt.subplots_adjust(wspace = 1.6)
plt.figure(figsize=(14,7))
plt.subplot(121)
application_train["TARGET"].value_counts().plot.pie(autopct = "%1.0f%%",colors = sns.color_palette("prism",7),startangle = 60,labels=["repayer","defaulter"],
                                                                        wedgeprops={"linewidth":2,"edgecolor":"k"},explode=[.1,0],shadow =True)
plt.title("Distribution of target variable")

plt.subplot(122)
ax = application_train["TARGET"].value_counts().plot(kind="barh")

for i,j in enumerate(application_train["TARGET"].value_counts().values):
    ax.text(.7,i,j,weight = "bold",fontsize=20)

plt.title("Count of target variable")
plt.show()
#Concatenating train and test data
application_train_x = application_train[[x for x in application_train.columns if x not in ["TARGET"]]]
application_train_x["type"] = "train"
application_test["type"]    = "test"
data = pd.concat([application_train_x,application_test],axis=0) 
plt.figure(figsize=(14,7))
plt.subplot(121)
data[data["type"] == "train"]["NAME_CONTRACT_TYPE"].value_counts().plot.pie(autopct = "%1.0f%%",colors = ["grey","orange"],startangle = 60,
                                                                        wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)
circ = plt.Circle((0,0),.7,color="white")
plt.gca().add_artist(circ)
plt.title("distribution of contract types in train data")

plt.subplot(122)
data[data["type"] == "test"]["NAME_CONTRACT_TYPE"].value_counts().plot.pie(autopct = "%1.0f%%",colors = ["grey","orange"],startangle = 60,
                                                                        wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)
circ = plt.Circle((0,0),.7,color="white")
plt.gca().add_artist(circ)
plt.ylabel("")
plt.title("distribution of contract types in test data")
plt.show()
fig = plt.figure(figsize=(13,6))
plt.subplot(121)
data[data["type"] == "train"]["CODE_GENDER"].value_counts().plot.pie(autopct = "%1.0f%%",colors = ["white","r"],startangle = 60,
                                                                        wedgeprops={"linewidth":2,"edgecolor":"k"},explode=[.05,0,0],shadow =True)
plt.title("distribution of gender in train data")

plt.subplot(122)
data[data["type"] == "test"]["CODE_GENDER"].value_counts().plot.pie(autopct = "%1.0f%%",colors = ["white","r"],startangle = 60,
                                                                        wedgeprops={"linewidth":2,"edgecolor":"k"},explode=[.05,0],shadow =True)
plt.ylabel("")
plt.title("distribution of gender in test data")
fig.set_facecolor("lightgrey")
fig  = plt.figure(figsize=(13,6))
plt.subplot(121)
ax = sns.countplot("NAME_CONTRACT_TYPE",hue="CODE_GENDER",data=data[data["type"] == "train"],palette=["r","b","g"])
ax.set_facecolor("lightgrey")
ax.set_title("Distribution of Contract type by gender -train data")

plt.subplot(122)
ax1 = sns.countplot("NAME_CONTRACT_TYPE",hue="CODE_GENDER",data=data[data["type"] == "test"],palette=["b","r"])
ax1.set_facecolor("lightgrey")
ax1.set_title("Distribution of Contract type by gender -test data")
plt.show()
fig = plt.figure(figsize=(13,6))

plt.subplot(121)
data["FLAG_OWN_CAR"].value_counts().plot.pie(autopct = "%1.0f%%",colors = ["gold","orangered"],startangle = 60,
                                                                        wedgeprops={"linewidth":2,"edgecolor":"k"},explode=[.05,0],shadow =True)
plt.title("distribution of client owning a car")

plt.subplot(122)
data[data["FLAG_OWN_CAR"] == "Y"]["CODE_GENDER"].value_counts().plot.pie(autopct = "%1.0f%%",colors = ["b","orangered"],startangle = 90,
                                                                        wedgeprops={"linewidth":2,"edgecolor":"k"},explode=[.05,0,0],shadow =True)
plt.title("distribution of client owning a car by gender")

plt.show()
plt.figure(figsize=(13,6))
plt.subplot(121)
data["FLAG_OWN_REALTY"].value_counts().plot.pie(autopct = "%1.0f%%",colors = ["skyblue","gold"],startangle = 90,
                                              wedgeprops={"linewidth":2,"edgecolor":"k"},explode=[0.05,0],shadow =True)
plt.title("Distribution of client owns a house or flat")

plt.subplot(122)
data[data["FLAG_OWN_REALTY"] == "Y"]["CODE_GENDER"].value_counts().plot.pie(autopct = "%1.0f%%",colors = ["orangered","b"],startangle = 90,
                                                                        wedgeprops={"linewidth":2,"edgecolor":"k"},explode=[.05,0,0],shadow =True)
plt.title("Distribution of client owning a house or flat by gender")
plt.show()
fig = plt.figure(figsize=(12,10))
plt.subplot(211)
sns.countplot(application_train["CNT_CHILDREN"],palette="Set1",hue=application_train["TARGET"])
plt.legend(loc="upper center")
plt.title(" Distribution of Number of children client has  by repayment status")
plt.subplot(212)
sns.countplot(application_train["CNT_FAM_MEMBERS"],palette="Set1",hue=application_train["TARGET"])
plt.legend(loc="upper center")
plt.title(" Distribution of Number of family members client has  by repayment status")
fig.set_facecolor("lightblue")
default = application_train[application_train["TARGET"]==1][[ 'NAME_CONTRACT_TYPE', 'CODE_GENDER','FLAG_OWN_CAR', 'FLAG_OWN_REALTY']]
non_default = application_train[application_train["TARGET"]==0][[ 'NAME_CONTRACT_TYPE', 'CODE_GENDER','FLAG_OWN_CAR', 'FLAG_OWN_REALTY']]

d_cols = ['NAME_CONTRACT_TYPE', 'CODE_GENDER','FLAG_OWN_CAR', 'FLAG_OWN_REALTY']
d_length = len(d_cols)

fig = plt.figure(figsize=(16,4))
for i,j in itertools.zip_longest(d_cols,range(d_length)):
    plt.subplot(1,4,j+1)
    default[i].value_counts().plot.pie(autopct = "%1.0f%%",colors = sns.color_palette("prism"),startangle = 90,
                                        wedgeprops={"linewidth":1,"edgecolor":"white"},shadow =True)
    circ = plt.Circle((0,0),.7,color="white")
    plt.gca().add_artist(circ)
    plt.ylabel("")
    plt.title(i+"-Defaulter")


fig = plt.figure(figsize=(16,4))
for i,j in itertools.zip_longest(d_cols,range(d_length)):
    plt.subplot(1,4,j+1)
    non_default[i].value_counts().plot.pie(autopct = "%1.0f%%",colors = sns.color_palette("prism",3),startangle = 90,
                                           wedgeprops={"linewidth":1,"edgecolor":"white"},shadow =True)
    circ = plt.Circle((0,0),.7,color="white")
    plt.gca().add_artist(circ)
    plt.ylabel("")
    plt.title(i+"-Repayer")

cols = [ 'AMT_INCOME_TOTAL', 'AMT_CREDIT','AMT_ANNUITY', 'AMT_GOODS_PRICE']
length = len(cols)
cs = ["r","b","g","k"]

ax = plt.figure(figsize=(12,12))
ax.set_facecolor("lightgrey")
for i,j,k in itertools.zip_longest(cols,range(length),cs):
    plt.subplot(2,2,j+1)
    sns.distplot(data[data[i].notnull()][i],color=k)
    plt.axvline(data[i].mean(),label = "mean",linestyle="dashed",color="k")
    plt.legend(loc="best")
    plt.title(i)
    plt.subplots_adjust(hspace = .2)
cols = [ 'AMT_INCOME_TOTAL', 'AMT_CREDIT','AMT_ANNUITY', 'AMT_GOODS_PRICE']

df = application_train.groupby("TARGET")[cols].describe().transpose().reset_index()
df = df[df["level_1"].isin([ 'mean', 'std', 'min', 'max'])] 
df_x = df[["level_0","level_1",0]]
df_y = df[["level_0","level_1",1]]
df_x = df_x.rename(columns={'level_0':"amount_type", 'level_1':"statistic", 0:"amount"})
df_x["type"] = "REPAYER"
df_y = df_y.rename(columns={'level_0':"amount_type", 'level_1':"statistic", 1:"amount"})
df_y["type"] = "DEFAULTER"
df_new = pd.concat([df_x,df_y],axis = 0)

stat = df_new["statistic"].unique().tolist()
length = len(stat)

plt.figure(figsize=(13,15))

for i,j in itertools.zip_longest(stat,range(length)):
    plt.subplot(2,2,j+1)
    fig = sns.barplot(df_new[df_new["statistic"] == i]["amount_type"],df_new[df_new["statistic"] == i]["amount"],
                hue=df_new[df_new["statistic"] == i]["type"],palette=["g","r"])
    plt.title(i + "--Defaulters vs Non defaulters")
    plt.subplots_adjust(hspace = .4)
    fig.set_facecolor("lightgrey")
cols = [ 'AMT_INCOME_TOTAL', 'AMT_CREDIT','AMT_ANNUITY', 'AMT_GOODS_PRICE']

df1 = data.groupby("CODE_GENDER")[cols].mean().transpose().reset_index()

df_f   = df1[["index","F"]]
df_f   = df_f.rename(columns={'index':"amt_type", 'F':"amount"})
df_f["gender"] = "FEMALE"
df_m   = df1[["index","M"]]
df_m   = df_m.rename(columns={'index':"amt_type", 'M':"amount"})
df_m["gender"] = "MALE"
df_xna = df1[["index","XNA"]]
df_xna = df_xna.rename(columns={'index':"amt_type", 'XNA':"amount"})
df_xna["gender"] = "XNA"

df_gen = pd.concat([df_m,df_f,df_xna],axis=0)

plt.figure(figsize=(12,5))
ax = sns.barplot("amt_type","amount",data=df_gen,hue="gender",palette="Set1")
plt.title("Average Income,credit,annuity & goods_price by gender")
plt.show()
fig = plt.figure(figsize=(10,8))
plt.scatter(application_train[application_train["TARGET"]==0]['AMT_ANNUITY'],application_train[application_train["TARGET"]==0]['AMT_CREDIT'],s=35,
            color="b",alpha=.5,label="REPAYER",linewidth=.5,edgecolor="k")
plt.scatter(application_train[application_train["TARGET"]==1]['AMT_ANNUITY'],application_train[application_train["TARGET"]==1]['AMT_CREDIT'],s=35,
            color="r",alpha=.2,label="DEFAULTER",linewidth=.5,edgecolor="k")
plt.legend(loc="best",prop={"size":15})
plt.xlabel("AMT_ANNUITY")
plt.ylabel("AMT_CREDIT")
plt.title("Scatter plot between credit amount and annuity amount")
plt.show()
amt = application_train[[ 'AMT_INCOME_TOTAL','AMT_CREDIT',
                         'AMT_ANNUITY', 'AMT_GOODS_PRICE',"TARGET"]]
amt = amt[(amt["AMT_GOODS_PRICE"].notnull()) & (amt["AMT_ANNUITY"].notnull())]
sns.pairplot(amt,hue="TARGET",palette=["b","r"])
plt.show()
plt.figure(figsize=(12,5))
plt.subplot(121)
sns.countplot(y=data["NAME_TYPE_SUITE"],
              palette="Set2",
              order=data["NAME_TYPE_SUITE"].value_counts().index[:5])
plt.title("Distribution of Suite type")

plt.subplot(122)
sns.countplot(y=data["NAME_TYPE_SUITE"],
              hue=data["CODE_GENDER"],palette="Set2",
              order=data["NAME_TYPE_SUITE"].value_counts().index[:5])
plt.ylabel("")
plt.title("Distribution of Suite type by gender")
plt.subplots_adjust(wspace = .4)
plt.figure(figsize=(12,5))
plt.subplot(121)
sns.countplot(y=data["NAME_INCOME_TYPE"],
              palette="Set2",
              order=data["NAME_INCOME_TYPE"].value_counts().index[:4])
plt.title("Distribution of client income type")

plt.subplot(122)
sns.countplot(y=data["NAME_INCOME_TYPE"],
              hue=data["CODE_GENDER"],
              palette="Set2",
              order=data["NAME_INCOME_TYPE"].value_counts().index[:4])
plt.ylabel("")
plt.title("Distribution of client income  type by gender")
plt.subplots_adjust(wspace = .4)
plt.figure(figsize=(16,8))
plt.subplot(121)
application_train[application_train["TARGET"]==0]["NAME_EDUCATION_TYPE"].value_counts().plot.pie(fontsize=9,autopct = "%1.0f%%",
                                                                                                 colors = sns.color_palette("Set1"),
                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)
circ = plt.Circle((0,0),.7,color="white")
plt.gca().add_artist(circ)
plt.title("Distribution of Education type for Repayers",color="b")

plt.subplot(122)
application_train[application_train["TARGET"]==1]["NAME_EDUCATION_TYPE"].value_counts().plot.pie(fontsize=9,autopct = "%1.0f%%",
                                                                                                 colors = sns.color_palette("Set1"),
                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)
circ = plt.Circle((0,0),.7,color="white")
plt.gca().add_artist(circ)
plt.title("Distribution of Education type for Defaulters",color="b")
plt.ylabel("")
plt.show()
edu = data.groupby(['NAME_EDUCATION_TYPE','NAME_INCOME_TYPE'])['AMT_INCOME_TOTAL'].mean().reset_index().sort_values(by='AMT_INCOME_TOTAL',ascending=False)
fig = plt.figure(figsize=(13,7))
ax = sns.barplot('NAME_INCOME_TYPE','AMT_INCOME_TOTAL',data=edu,hue='NAME_EDUCATION_TYPE',palette="seismic")
ax.set_facecolor("k")
plt.title(" Average Earnings by different professions and education types")
plt.show()
plt.figure(figsize=(16,8))
plt.subplot(121)
application_train[application_train["TARGET"]==0]["NAME_FAMILY_STATUS"].value_counts().plot.pie(autopct = "%1.0f%%",
                                                             startangle=120,colors = sns.color_palette("Set2",7),
                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True,explode=[0,.07,0,0,0,0])

plt.title("Distribution of Family status for Repayers",color="b")

plt.subplot(122)
application_train[application_train["TARGET"]==1]["NAME_FAMILY_STATUS"].value_counts().plot.pie(autopct = "%1.0f%%",
                                                    startangle=120,colors = sns.color_palette("Set2",7),
                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True,explode=[0,.07,0,0,0])


plt.title("Distribution of Family status for Defaulters",color="b")
plt.ylabel("")
plt.show()
plt.figure(figsize=(16,8))
plt.subplot(121)
application_train[application_train["TARGET"]==0]["NAME_HOUSING_TYPE"].value_counts().plot.pie(autopct = "%1.0f%%",fontsize=8,
                                                             colors = sns.color_palette("Spectral"),
                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)

plt.title("Distribution of housing type  for Repayer",color="b")

plt.subplot(122)
application_train[application_train["TARGET"]==1]["NAME_HOUSING_TYPE"].value_counts().plot.pie(autopct = "%1.0f%%",fontsize=8,
                                                    colors = sns.color_palette("Spectral"),
                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)


plt.title("Distribution of housing type for Defaulters",color="b")
plt.ylabel("")
plt.show()
fig = plt.figure(figsize=(13,8))

plt.subplot(121)
sns.violinplot(y=application_train[application_train["TARGET"]==0]["REGION_POPULATION_RELATIVE"]
               ,x=application_train[application_train["TARGET"]==0]["NAME_CONTRACT_TYPE"],
               palette="Set1")
plt.title("Distribution of region population for Non Default loans",color="b")
plt.subplot(122)
sns.violinplot(y = application_train[application_train["TARGET"]==1]["REGION_POPULATION_RELATIVE"]
               ,x=application_train[application_train["TARGET"]==1]["NAME_CONTRACT_TYPE"]
               ,palette="Set1")
plt.title("Distribution of region population  for  Default loans",color="b")

plt.subplots_adjust(wspace = .2)
fig.set_facecolor("lightgrey")
fig = plt.figure(figsize=(13,15))

plt.subplot(221)
sns.distplot(application_train[application_train["TARGET"]==0]["DAYS_BIRTH"],color="b")
plt.title("Age Distribution of repayers")

plt.subplot(222)
sns.distplot(application_train[application_train["TARGET"]==1]["DAYS_BIRTH"],color="r")
plt.title("Age Distribution of defaulters")

plt.subplot(223)
sns.lvplot(application_train["TARGET"],application_train["DAYS_BIRTH"],hue=application_train["CODE_GENDER"],palette=["b","grey","m"])
plt.axhline(application_train["DAYS_BIRTH"].mean(),linestyle="dashed",color="k",label ="average age of client")
plt.legend(loc="lower right")
plt.title("Client age vs Loan repayment status(hue=gender)")

plt.subplot(224)
sns.lvplot(application_train["TARGET"],application_train["DAYS_BIRTH"],hue=application_train["NAME_CONTRACT_TYPE"],palette=["r","g"])
plt.axhline(application_train["DAYS_BIRTH"].mean(),linestyle="dashed",color="k",label ="average age of client")
plt.legend(loc="lower right")
plt.title("Client age vs Loan repayment status(hue=contract type)")

plt.subplots_adjust(wspace = .2,hspace = .3)

fig.set_facecolor("lightgrey")
fig = plt.figure(figsize=(13,5))

plt.subplot(121)
sns.distplot(application_train[application_train["TARGET"]==0]["DAYS_EMPLOYED"],color="b")
plt.title("days employed distribution of repayers")

plt.subplot(122)
sns.distplot(application_train[application_train["TARGET"]==1]["DAYS_EMPLOYED"],color="r")
plt.title("days employed distribution of defaulters")

fig.set_facecolor("ghostwhite")
fig = plt.figure(figsize=(13,5))

plt.subplot(121)
sns.distplot(application_train[application_train["TARGET"]==0]["DAYS_REGISTRATION"],color="b")
plt.title("registration days distribution of repayers")

plt.subplot(122)
sns.distplot(application_train[application_train["TARGET"]==1]["DAYS_REGISTRATION"],color="r")
plt.title("registration days distribution of defaulter")

fig.set_facecolor("ghostwhite")
fig = plt.figure(figsize=(15,7))
plt.subplot(121)
sns.violinplot(y = application_train[application_train["OWN_CAR_AGE"].notnull()]["OWN_CAR_AGE"],
               x=application_train[application_train["OWN_CAR_AGE"].notnull()]["TARGET"])
plt.axhline(application_train[(application_train["OWN_CAR_AGE"].notnull())&(application_train["TARGET"] ==0)]["OWN_CAR_AGE"].mean(),color="b",
            linestyle="dashed",label = "average car age of repayers")
plt.axhline(application_train[(application_train["OWN_CAR_AGE"].notnull())&(application_train["TARGET"] ==1)]["OWN_CAR_AGE"].mean(),color="r",
            linestyle="dashed",label = "average car age of defaulters")
plt.legend(loc="best")
plt.title("Distribution of car age by repayment status")

plt.subplot(122)
sns.distplot(application_train[application_train["OWN_CAR_AGE"].notnull()]["OWN_CAR_AGE"],color="k")
plt.title("Distribution of car age")

fig.set_facecolor("lightgrey")
x   = application_train[['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE',
       'FLAG_PHONE', 'FLAG_EMAIL',"TARGET"]]
x["TARGET"] = x["TARGET"].replace({0:"repayers",1:"defaulters"})
x  = x.replace({1:"YES",0:"NO"})

cols = ['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE',
       'FLAG_PHONE', 'FLAG_EMAIL']
length = len(cols)

fig = plt.figure(figsize=(15,12))
fig.set_facecolor("lightgrey")

for i,j in itertools.zip_longest(cols,range(length)):
    plt.subplot(2,3,j+1)
    sns.countplot(x[i],hue=x["TARGET"],palette=["r","g"])
    plt.title(i,color="b")

fig = plt.figure(figsize=(13,7))
occ = application_train[application_train["TARGET"]==0]["OCCUPATION_TYPE"].value_counts().reset_index()
occ = occ.sort_values(by = "index",ascending=True)
occ1 = application_train[application_train["TARGET"]==1]["OCCUPATION_TYPE"].value_counts().reset_index()
occ1 = occ1.sort_values(by = "index",ascending=True)
occ["percentage"]  = (occ["OCCUPATION_TYPE"]*100/occ["OCCUPATION_TYPE"].sum())
occ1["percentage"] = (occ1["OCCUPATION_TYPE"]*100/occ1["OCCUPATION_TYPE"].sum())
occ["type"]        = "Repayers"
occ1["type"]       = "defaulters"
occupation = pd.concat([occ,occ1],axis=0)

ax = sns.barplot("index","percentage",data=occupation,hue="type",palette=["b","r"])
plt.xticks(rotation = 70)
plt.xlabel("occupation")
ax.set_facecolor("k")
fig.set_facecolor("ghostwhite")
plt.title("Occupation percentage in data with respect to repayment status")
plt.show()
fig = plt.figure(figsize=(13,13))
plt.subplot(221)
application_train[application_train["TARGET"]==0]["REGION_RATING_CLIENT"].value_counts().plot.pie(autopct = "%1.0f%%",fontsize=8,
                                                             colors = sns.color_palette("Pastel1"),
                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)

plt.title("Distribution of region rating  for Repayers",color="b")

plt.subplot(222)
application_train[application_train["TARGET"]==1]["REGION_RATING_CLIENT"].value_counts().plot.pie(autopct = "%1.0f%%",fontsize=8,
                                                    colors = sns.color_palette("Pastel1"),
                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)


plt.title("Distribution of region rating  for Defaulters",color="b")
plt.ylabel("")

plt.subplot(223)
application_train[application_train["TARGET"]==0]["REGION_RATING_CLIENT_W_CITY"].value_counts().plot.pie(autopct = "%1.0f%%",fontsize=8,
                                                             colors = sns.color_palette("Paired"),
                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)

plt.title("Distribution of city region rating   for Repayers",color="b")

plt.subplot(224)
application_train[application_train["TARGET"]==1]["REGION_RATING_CLIENT_W_CITY"].value_counts().plot.pie(autopct = "%1.0f%%",fontsize=8,
                                                    colors = sns.color_palette("Paired"),
                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)


plt.title("Distribution of city region rating  for Defaulters",color="b")
plt.ylabel("")
fig.set_facecolor("ivory")
day = application_train.groupby("TARGET").agg({"WEEKDAY_APPR_PROCESS_START":"value_counts"})
day = day.rename(columns={"WEEKDAY_APPR_PROCESS_START":"value_counts"})
day = day.reset_index()
day_0 = day[:7]
day_1 = day[7:]
day_0["percentage"] = day_0["value_counts"]*100/day_0["value_counts"].sum()
day_1["percentage"] = day_1["value_counts"]*100/day_1["value_counts"].sum()
days = pd.concat([day_0,day_1],axis=0)
days["TARGET"] = days.replace({1:"defaulters",0:"repayers"})

fig = plt.figure(figsize=(13,15))
plt.subplot(211)
order = ['SUNDAY', 'MONDAY','TUESDAY', 'WEDNESDAY','THURSDAY', 'FRIDAY', 'SATURDAY']
ax= sns.barplot("WEEKDAY_APPR_PROCESS_START","percentage",data=days,
                hue="TARGET",order=order,palette="prism")
ax.set_facecolor("k")
ax.set_title("Peak days for applying loans (defaulters vs repayers)")

hr = application_train.groupby("TARGET").agg({"HOUR_APPR_PROCESS_START":"value_counts"})
hr = hr.rename(columns={"HOUR_APPR_PROCESS_START":"value_counts"}).reset_index()
hr_0 = hr[hr["TARGET"]==0]
hr_1 = hr[hr["TARGET"]==1]
hr_0["percentage"] = hr_0["value_counts"]*100/hr_0["value_counts"].sum()
hr_1["percentage"] = hr_1["value_counts"]*100/hr_1["value_counts"].sum()
hrs = pd.concat([hr_0,hr_1],axis=0)
hrs["TARGET"] = hrs["TARGET"].replace({1:"defaulters",0:"repayers"})
hrs = hrs.sort_values(by="HOUR_APPR_PROCESS_START",ascending=True)

plt.subplot(212)
ax1 = sns.pointplot("HOUR_APPR_PROCESS_START","percentage",
                    data=hrs,hue="TARGET",palette="prism")
ax1.set_facecolor("k")
ax1.set_title("Peak hours for applying loans (defaulters vs repayers)")
fig.set_facecolor("snow")
org = application_train.groupby("TARGET").agg({"ORGANIZATION_TYPE":"value_counts"})
org = org.rename(columns = {"ORGANIZATION_TYPE":"value_counts"}).reset_index()
org_0 = org[org["TARGET"] == 0]
org_1 = org[org["TARGET"] == 1]
org_0["percentage"] = org_0["value_counts"]*100/org_0["value_counts"].sum()
org_1["percentage"] = org_1["value_counts"]*100/org_1["value_counts"].sum()

organization = pd.concat([org_0,org_1],axis=0)
organization = organization.sort_values(by="ORGANIZATION_TYPE",ascending=True)

organization["TARGET"] = organization["TARGET"].replace({0:"repayers",1:"defaulters"})

organization
plt.figure(figsize=(13,7))
ax = sns.pointplot("ORGANIZATION_TYPE","percentage",
                   data=organization,hue="TARGET",palette=["b","r"])
plt.xticks(rotation=90)
plt.grid(True,alpha=.3)
ax.set_facecolor("k")
ax.set_title("Distribution in organization types for repayers and defaulters")
plt.show()
application_train[["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3"]]

fig = plt.figure(figsize=(13,20))

plt.subplot(321)
sns.distplot(application_train[(application_train["EXT_SOURCE_1"].notnull()) & (application_train["TARGET"] ==0 )]["EXT_SOURCE_1"],color="b")
plt.axvline(application_train[(application_train["EXT_SOURCE_1"].notnull()) & (application_train["TARGET"] ==0 )]["EXT_SOURCE_1"].mean(),
           linestyle="dashed",color="k",
           label = ["MEAN :",application_train[(application_train["EXT_SOURCE_1"].notnull()) & (application_train["TARGET"] ==0 )]["EXT_SOURCE_1"].mean()])
plt.legend(loc="best")
plt.title("Repayer EXT_SOURCE_1 distribution")

plt.subplot(322)
sns.distplot(application_train[(application_train["EXT_SOURCE_1"].notnull()) & (application_train["TARGET"] ==1 )]["EXT_SOURCE_1"],color="r")
plt.axvline(application_train[(application_train["EXT_SOURCE_1"].notnull()) & (application_train["TARGET"] ==1 )]["EXT_SOURCE_1"].mean(),
           linestyle="dashed",color="k",
           label = ["MEAN :",application_train[(application_train["EXT_SOURCE_1"].notnull()) & (application_train["TARGET"] ==1 )]["EXT_SOURCE_1"].mean()])
plt.legend(loc="best")
plt.title("Defaulter EXT_SOURCE_1 distribution")
####
plt.subplot(323)
sns.distplot(application_train[(application_train["EXT_SOURCE_2"].notnull()) & (application_train["TARGET"] ==0 )]["EXT_SOURCE_2"],color="b")
plt.axvline(application_train[(application_train["EXT_SOURCE_2"].notnull()) & (application_train["TARGET"] ==0 )]["EXT_SOURCE_2"].mean(),
           linestyle="dashed",color="k",
           label = ["MEAN :",application_train[(application_train["EXT_SOURCE_2"].notnull()) & (application_train["TARGET"] ==0 )]["EXT_SOURCE_2"].mean()])
plt.legend(loc="best")
plt.title("Repayer EXT_SOURCE_2 distribution")

plt.subplot(324)
sns.distplot(application_train[(application_train["EXT_SOURCE_2"].notnull()) & (application_train["TARGET"] ==1 )]["EXT_SOURCE_2"],color="r")
plt.axvline(application_train[(application_train["EXT_SOURCE_2"].notnull()) & (application_train["TARGET"] ==1 )]["EXT_SOURCE_2"].mean(),
           linestyle="dashed",color="k",
           label = ["MEAN :",application_train[(application_train["EXT_SOURCE_2"].notnull()) & (application_train["TARGET"] ==1 )]["EXT_SOURCE_2"].mean()])
plt.legend(loc="best")
plt.title("Defaulter EXT_SOURCE_2 distribution")

###
plt.subplot(325)
sns.distplot(application_train[(application_train["EXT_SOURCE_3"].notnull()) & (application_train["TARGET"] ==0 )]["EXT_SOURCE_3"],color="b")
plt.axvline(application_train[(application_train["EXT_SOURCE_3"].notnull()) & (application_train["TARGET"] ==0 )]["EXT_SOURCE_3"].mean(),
           linestyle="dashed",color="k",
           label = ["MEAN :",application_train[(application_train["EXT_SOURCE_3"].notnull()) & (application_train["TARGET"] ==0 )]["EXT_SOURCE_3"].mean()])
plt.legend(loc="best")
plt.title("Repayer EXT_SOURCE_3 distribution")

plt.subplot(326)
sns.distplot(application_train[(application_train["EXT_SOURCE_3"].notnull()) & (application_train["TARGET"] ==1 )]["EXT_SOURCE_3"],color="r")
plt.axvline(application_train[(application_train["EXT_SOURCE_3"].notnull()) & (application_train["TARGET"] ==1 )]["EXT_SOURCE_3"].mean(),
           linestyle="dashed",color="k",
           label = ["MEAN :",application_train[(application_train["EXT_SOURCE_3"].notnull()) & (application_train["TARGET"] ==1 )]["EXT_SOURCE_3"].mean()])
plt.legend(loc="best")
plt.title("Defaulter EXT_SOURCE_3 distribution")

plt.subplots_adjust(hspace = .3)
fig.set_facecolor("lightgrey")
fig = plt.figure(figsize=(12,13))

cols = ['APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG',
       'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG',
       'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG',
       'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG',
       'NONLIVINGAREA_AVG']

plt.subplot(211)
sns.heatmap(application_train[application_train["TARGET"] == 0][cols].describe()[1:].transpose(),
            annot=True,cmap=sns.color_palette("Set1"),
            linecolor="k",linewidth=1)
plt.title("descriptive stats for Average  Normalized information about building where the repayers lives.",color="b")

plt.subplot(212)
sns.heatmap(application_train[application_train["TARGET"] == 1][cols].describe()[1:].transpose(),
            annot=True,cmap=sns.color_palette("Set1"),
           linecolor="k",linewidth=1)
plt.title("descriptive stats for Average  Normalized information about building where the defaulters lives.",color="b")
fig.set_facecolor("ghostwhite")
fig = plt.figure(figsize=(13,13))

cols1 = ['APARTMENTS_MODE', 'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE',
       'YEARS_BUILD_MODE', 'COMMONAREA_MODE', 'ELEVATORS_MODE',
       'ENTRANCES_MODE', 'FLOORSMAX_MODE', 'FLOORSMIN_MODE', 'LANDAREA_MODE',
       'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_MODE']

plt.subplot(211)
sns.heatmap(application_train[application_train["TARGET"] == 0][cols1].describe()[1:].transpose(),annot=True,cmap=sns.color_palette("viridis"),
            linecolor="k",linewidth=1)
plt.title("descriptive stats for Mode  Normalized information about building where the repayers lives.",color="b")

plt.subplot(212)
sns.heatmap(application_train[application_train["TARGET"] == 1][cols1].describe()[1:].transpose(),annot=True,cmap=sns.color_palette("viridis"),
           linecolor="k",linewidth=1)
plt.title("descriptive stats for Mode  Normalized information about building where the defaulters lives.",color="b")
fig.set_facecolor("ghostwhite")
fig = plt.figure(figsize=(12,13))

cols2 = ['APARTMENTS_MEDI', 'BASEMENTAREA_MEDI',
       'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI',
       'ELEVATORS_MEDI', 'ENTRANCES_MEDI', 'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI',
       'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI', 'LIVINGAREA_MEDI',
       'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI']

plt.subplot(211)
sns.heatmap(application_train[application_train["TARGET"] == 0][cols2].describe()[1:].transpose(),
            annot=True,cmap=sns.color_palette("magma"),
            linecolor="k",linewidth=1)
plt.title("descriptive stats for Median Normalized information about building where the repayers lives.",color="b")

plt.subplot(212)
sns.heatmap(application_train[application_train["TARGET"] == 1][cols2].describe()[1:].transpose(),
            annot=True,cmap=sns.color_palette("magma"),
           linecolor="k",linewidth=1)
plt.title("descriptive stats for Median  Normalized information about building where the defaulters lives.",color="b")
fig.set_facecolor("ghostwhite")
col = cols + cols1 +cols2

avg_mean = application_train.groupby("TARGET")[col].mean().stack().reset_index()
avg_mean["TARGET"] = avg_mean["TARGET"].replace({1:"defaulters",0:"repayers"})
avg_std = application_train.groupby("TARGET")[col].std().stack().reset_index()
avg_std["TARGET"] = avg_std["TARGET"].replace({1:"defaulters",0:"repayers"})

fig =plt.figure(figsize=(14,15))
plt.subplot(211)
ax = sns.barplot("level_1",0,data=avg_mean,hue="TARGET",palette=["white","k"])
plt.xticks(rotation=90)
plt.xlabel("")
plt.ylabel("mean")
plt.title("comparing mean values of normalized values between repayers and defaulters",color="b")
ax.set_facecolor("r")

plt.subplot(212)
ax1 = sns.pointplot("level_1",0,data=avg_std,hue="TARGET",palette=["white","k"])
plt.xticks(rotation=90)
plt.xlabel("")
plt.ylabel("standard deviation")
plt.title("comparing standard deviation of normalized values between repayers and defaulters",color="b")
ax1.set_facecolor("r")
plt.subplots_adjust(hspace = .7)

import squarify
cs = [ 'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE']
length = len(cs)
pal = ["Set1","Set3","Set2"]

plt.figure(figsize=(13,10))
for i,j,k in itertools.zip_longest(cs,range(length),pal):
    plt.subplot(2,2,j+1)
    squarify.plot(sizes=data[i].value_counts().values,label=data[i].value_counts().keys(),
                  value=data[i].value_counts().values,
                  color=sns.color_palette(k),linewidth=2,edgecolor="k",alpha=.8)
    plt.title(i)
    plt.axis("off")
cs = [ 'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE']
length = len(cs)

fig = plt.figure(figsize=(13,14))
fig.set_facecolor("lightgrey")
for i,j in itertools.zip_longest(cs,range(length)):
    plt.subplot(2,2,j+1)
    ax = sns.barplot(i,"TOTALAREA_MODE",data=application_train.groupby(["TARGET",i])["TOTALAREA_MODE"].mean().reset_index(),
                hue="TARGET",palette=["b","r"])
    ax.set_facecolor("yellow")
    plt.title(i)
fig = plt.figure(figsize=(13,20))
plt.subplot(421)
ax = sns.distplot(application_train[(application_train["TARGET"] == 1 ) & (application_train["OBS_30_CNT_SOCIAL_CIRCLE"].notnull())]["OBS_30_CNT_SOCIAL_CIRCLE"],
                color="r",label="defaulter")
plt.axvline(application_train[(application_train["TARGET"] == 1 ) ]["OBS_30_CNT_SOCIAL_CIRCLE"].mean(),label="mean",color="k",linestyle="dashed")
plt.legend(loc="best")
plt.title("OBS_30_CNT_SOCIAL_CIRCLE - defaulter")
plt.xlabel("")
ax.set_facecolor("lightyellow")

plt.subplot(422)
ax = sns.distplot(application_train[(application_train["TARGET"] == 0 ) & (application_train["OBS_30_CNT_SOCIAL_CIRCLE"].notnull())]["OBS_30_CNT_SOCIAL_CIRCLE"],
                color="b",label="repayer")
plt.axvline(application_train[(application_train["TARGET"] == 0 ) ]["OBS_30_CNT_SOCIAL_CIRCLE"].mean(),label="mean",color="k",linestyle="dashed")
plt.legend(loc="best")
plt.title("OBS_30_CNT_SOCIAL_CIRCLE - repayer")
plt.xlabel("")
ax.set_facecolor("lightyellow")

plt.subplot(423)
ax = sns.distplot(application_train[(application_train["TARGET"] == 1 ) & (application_train["DEF_30_CNT_SOCIAL_CIRCLE"].notnull())]["DEF_30_CNT_SOCIAL_CIRCLE"],
                color="r",label="defaulter")
plt.axvline(application_train[(application_train["TARGET"] == 1 ) ]["DEF_30_CNT_SOCIAL_CIRCLE"].mean(),label="mean",color="k",linestyle="dashed")
plt.legend(loc="best")
plt.title("DEF_30_CNT_SOCIAL_CIRCLE - defaulter")
plt.xlabel("")
ax.set_facecolor("lightyellow")

plt.subplot(424)
ax = sns.distplot(application_train[(application_train["TARGET"] == 0 ) & (application_train["DEF_30_CNT_SOCIAL_CIRCLE"].notnull())]["DEF_30_CNT_SOCIAL_CIRCLE"],
                color="b",label="repayer")
plt.axvline(application_train[(application_train["TARGET"] == 0 ) ]["DEF_30_CNT_SOCIAL_CIRCLE"].mean(),label="mean",color="k",linestyle="dashed")
plt.legend(loc="best")
plt.title("DEF_30_CNT_SOCIAL_CIRCLE - repayer")
plt.xlabel("")
ax.set_facecolor("lightyellow")

plt.subplot(425)
ax = sns.distplot(application_train[(application_train["TARGET"] == 1 ) & (application_train["OBS_60_CNT_SOCIAL_CIRCLE"].notnull())]["OBS_60_CNT_SOCIAL_CIRCLE"],
                color="r",label="defaulter")
plt.axvline(application_train[(application_train["TARGET"] == 1 ) ]["OBS_60_CNT_SOCIAL_CIRCLE"].mean(),label="mean",color="k",linestyle="dashed")
plt.legend(loc="best")
plt.title("OBS_60_CNT_SOCIAL_CIRCLE - defaulter")
plt.xlabel("")
ax.set_facecolor("lightyellow")

plt.subplot(426)
ax = sns.distplot(application_train[(application_train["TARGET"] == 0 ) & (application_train["OBS_60_CNT_SOCIAL_CIRCLE"].notnull())]["OBS_60_CNT_SOCIAL_CIRCLE"],
                color="b",label="repayer")
plt.axvline(application_train[(application_train["TARGET"] == 0 ) ]["OBS_60_CNT_SOCIAL_CIRCLE"].mean(),label="mean",color="k",linestyle="dashed")
plt.legend(loc="best")
plt.title("OBS_60_CNT_SOCIAL_CIRCLE - repayer")
plt.xlabel("")
ax.set_facecolor("lightyellow")



plt.subplot(427)
ax = sns.distplot(application_train[(application_train["TARGET"] == 1 ) & (application_train["DEF_60_CNT_SOCIAL_CIRCLE"].notnull())]["DEF_60_CNT_SOCIAL_CIRCLE"],
                color="r",label="defaulter")
plt.axvline(application_train[(application_train["TARGET"] == 1 ) ]["DEF_60_CNT_SOCIAL_CIRCLE"].mean(),label="mean",color="k",linestyle="dashed")
plt.legend(loc="best")
plt.title("DEF_60_CNT_SOCIAL_CIRCLE - defaulter")
plt.xlabel("")
ax.set_facecolor("lightyellow")

plt.subplot(428)

ax = sns.distplot(application_train[(application_train["TARGET"] == 0 ) & (application_train["DEF_60_CNT_SOCIAL_CIRCLE"].notnull())]["DEF_60_CNT_SOCIAL_CIRCLE"],
                color="b",label="repayer")
plt.axvline(application_train[(application_train["TARGET"] == 0 ) ]["DEF_60_CNT_SOCIAL_CIRCLE"].mean(),label="mean",color="k",linestyle="dashed")
plt.legend(loc="best")
plt.title("DEF_60_CNT_SOCIAL_CIRCLE - repayer")
plt.xlabel("")
ax.set_facecolor("lightyellow")
fig.set_facecolor("lightgrey")
plt.figure(figsize=(13,7))
plt.subplot(121)
ax = sns.violinplot(application_train["TARGET"],
                    application_train["DAYS_LAST_PHONE_CHANGE"],palette=["g","r"])
ax.set_facecolor("oldlace")
ax.set_title("days before application client changed phone -violin plot")
plt.subplot(122)
ax1 = sns.lvplot(application_train["TARGET"],
                 application_train["DAYS_LAST_PHONE_CHANGE"],palette=["g","r"])
ax1.set_facecolor("oldlace")
ax1.set_ylabel("")
ax1.set_title("days before application client changed phone -box plot")
plt.subplots_adjust(wspace = .2)
cols = [ 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3',
       'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6',
       'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9',
       'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12',
       'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15',
       'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18',
       'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']

df_flag = application_train[cols+["TARGET"]]

length = len(cols)

df_flag["TARGET"] = df_flag["TARGET"].replace({1:"defaulter",0:"repayer"})

fig = plt.figure(figsize=(13,24))
fig.set_facecolor("lightgrey")
for i,j in itertools.zip_longest(cols,range(length)):
    plt.subplot(5,4,j+1)
    ax = sns.countplot(df_flag[i],hue=df_flag["TARGET"],palette=["r","b"])
    plt.yticks(fontsize=5)
    plt.xlabel("")
    plt.title(i)
    ax.set_facecolor("k")

cols = ['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
       'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
       'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']
application_train.groupby("TARGET")[cols].max().transpose().plot(kind="barh",
                                                                 figsize=(10,5),width=.8)
plt.title("Maximum enquries made by defaulters and repayers")
application_train.groupby("TARGET")[cols].mean().transpose().plot(kind="barh",
                                                                  figsize=(10,5),width=.8)
plt.title("average enquries made by defaulters and repayers")
application_train.groupby("TARGET")[cols].std().transpose().plot(kind="barh",
                                                                 figsize=(10,5),width=.8)
plt.title("standard deviation in enquries made by defaulters and repayers")
plt.show()
#Merging bureau andapplication train data
app_tar = application_train[["TARGET","SK_ID_CURR"]]
app_bureau = bureau.merge(app_tar,left_on="SK_ID_CURR",right_on="SK_ID_CURR",how="left")


prev_cre = app_bureau.groupby(["TARGET","SK_ID_CURR"])["SK_ID_BUREAU"].nunique().reset_index()
fig = plt.figure(figsize=(12,12))
plt.subplot(211)
sns.distplot(prev_cre[prev_cre["TARGET"]==0]["SK_ID_BUREAU"],color="g")
plt.axvline(prev_cre[prev_cre["TARGET"]==0]["SK_ID_BUREAU"].mean(),linestyle="dashed",color="k",label = "mean")
plt.title("Number of previous credits by repayers")
plt.legend(loc="best")

plt.subplot(212)
sns.distplot(prev_cre[prev_cre["TARGET"]==1]["SK_ID_BUREAU"],color="r")
plt.axvline(prev_cre[prev_cre["TARGET"]==1]["SK_ID_BUREAU"].mean(),linestyle="dashed",color="k",label = "mean")
plt.legend(loc="best")
plt.title("Number of previous credits by defaulters")
fig.set_facecolor("lightgrey")
plt.figure(figsize=(12,6))
plt.subplot(121)
app_bureau[app_bureau["TARGET"]  ==0]["CREDIT_ACTIVE"].value_counts().plot.pie(autopct = "%1.1f%%",fontsize=8,
                                                             colors = sns.color_palette("Set1"),
                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)
plt.title("previous credit status for repayers")

plt.subplot(122)
app_bureau[app_bureau["TARGET"]  ==1]["CREDIT_ACTIVE"].value_counts().plot.pie(autopct = "%1.0f%%",fontsize=8,
                                                             colors = sns.color_palette("Set1"),
                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)
plt.title("previous credit status for defaulters")
plt.show()
plt.figure(figsize=(8,8))
app_bureau["CREDIT_CURRENCY"].value_counts().plot.pie(autopct = "%1.1f%%",fontsize=8,
                                                             colors = sns.color_palette("Set1"),
                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)
plt.ylabel("")
plt.title("distribution of currency of the Credit Bureau credit")
plt.show()
app_bureau["TARGET"] = app_bureau["TARGET"].replace({1:"defaulter",0:"repayer"})

plt.figure(figsize=(13,8))

plt.subplot(121)
sns.violinplot(y=app_bureau["DAYS_CREDIT"],x=app_bureau["TARGET"],palette="husl")
plt.title("days before current application client applied for bureau")

plt.subplot(122)
sns.boxplot(y=app_bureau["DAYS_CREDIT"],x=app_bureau["TARGET"],palette="husl")
plt.ylabel("")
plt.title("days before current application client applied for bureau")
plt.show()

types = ["defaulter","repayer"]
length = len(types)
cs = ["r","b"]

fig = plt.figure(figsize=(13,22))
plt.subplot(411)
for i,j,k in itertools.zip_longest(types,(range(length)),cs):
    ax = sns.kdeplot(app_bureau[(app_bureau["TARGET"]==i)&(app_bureau["CREDIT_DAY_OVERDUE"].notnull())]["CREDIT_DAY_OVERDUE"],shade=True,color=k,label=i)
    plt.axvline(app_bureau[(app_bureau["TARGET"]==i)&(app_bureau["CREDIT_DAY_OVERDUE"].notnull())]["CREDIT_DAY_OVERDUE"].mean(),color=k,linestyle="dashed",label="mean")
    plt.title("Number of days past due on Bureau credit at the time of application for related loan")
    ax.set_facecolor("lightgrey")
    
plt.subplot(412)
for i,j,k in itertools.zip_longest(types,(range(length)),cs):
    ax = sns.kdeplot(app_bureau[(app_bureau["TARGET"]==i)&(app_bureau["DAYS_CREDIT_ENDDATE"].notnull())]["DAYS_CREDIT_ENDDATE"],shade=True,color=k,label=i)
    plt.axvline(app_bureau[(app_bureau["TARGET"]==i)&(app_bureau["DAYS_CREDIT_ENDDATE"].notnull())]["DAYS_CREDIT_ENDDATE"].mean(),color=k,linestyle="dashed",label="mean")
    ax.set_facecolor("lightgrey")
    plt.title("Remaining duration of Bureau credit (in days) at the time of application in Home Credit")
    
plt.subplot(413)
for i,j,k in itertools.zip_longest(types,(range(length)),cs):
    ax = sns.kdeplot(app_bureau[(app_bureau["TARGET"]==i)&(app_bureau["DAYS_ENDDATE_FACT"].notnull())]["DAYS_ENDDATE_FACT"],shade=True,color=k,label=i)
    plt.axvline(app_bureau[(app_bureau["TARGET"]==i)&(app_bureau["DAYS_ENDDATE_FACT"].notnull())]["DAYS_ENDDATE_FACT"].mean(),color=k,linestyle="dashed",label="mean")
    ax.set_facecolor("lightgrey")
    plt.title("Days since Bureau credit ended at the time of application in Home Credit (only for closed credit")

plt.subplot(414)
for i,j,k in itertools.zip_longest(types,(range(length)),cs):
    ax = sns.kdeplot(app_bureau[(app_bureau["TARGET"]==i)&(app_bureau["DAYS_CREDIT_UPDATE"].notnull())]["DAYS_CREDIT_UPDATE"],shade=True,color=k,label=i)
    plt.axvline(app_bureau[(app_bureau["TARGET"]==i)&(app_bureau["DAYS_CREDIT_UPDATE"].notnull())]["DAYS_CREDIT_UPDATE"].mean(),color=k,linestyle="dashed",label="mean")
    ax.set_facecolor("lightgrey")
    plt.title("Number of days before loan application did last information about the Credit Bureau credit come.")

fig.set_facecolor("lightyellow")

rep = ((app_bureau[app_bureau["TARGET"] == "repayer"]["CREDIT_TYPE"].value_counts()*100)/app_bureau[app_bureau["TARGET"] == "repayer"]["CREDIT_TYPE"].value_counts().sum()).reset_index()
rep["type"] = "repayers"

defl = ((app_bureau[app_bureau["TARGET"] == "defaulter"]["CREDIT_TYPE"].value_counts()*100)/app_bureau[app_bureau["TARGET"] == "defaulter"]["CREDIT_TYPE"].value_counts().sum()).reset_index()
defl["type"] = "defaulters"

credit_types = pd.concat([rep,defl],axis=0)
credit_types = credit_types.sort_values(by="CREDIT_TYPE",ascending =False)
plt.figure(figsize=(10,8))
ax = sns.barplot("CREDIT_TYPE","index",data=credit_types[:10],hue="type",palette=["b","r"])
ax.set_ylabel("credit types")
ax.set_xlabel("percentage")
ax.set_facecolor("k")
ax.set_title('Type of Credit Bureau credit')
plt.show()
cols = ['AMT_CREDIT_MAX_OVERDUE', 'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT','AMT_CREDIT_SUM_OVERDUE']
plt.figure(figsize=(12,6))
sns.heatmap(app_bureau[cols].describe().transpose(),annot=True,
            linecolor="k",linewidth=2,cmap=sns.color_palette("Set1"))
plt.title("summary of amount variables")
plt.show()
d = app_bureau[(app_bureau["TARGET"]=="defaulter") & (app_bureau["CNT_CREDIT_PROLONG"].notnull())]["CNT_CREDIT_PROLONG"].value_counts()*100/app_bureau[(app_bureau["TARGET"]=="defaulter") & (app_bureau["CNT_CREDIT_PROLONG"].notnull())]["CNT_CREDIT_PROLONG"].value_counts().sum()
r = app_bureau[(app_bureau["TARGET"]=="repayer") & (app_bureau["CNT_CREDIT_PROLONG"].notnull())]["CNT_CREDIT_PROLONG"].value_counts()*100/app_bureau[(app_bureau["TARGET"]=="repayer") & (app_bureau["CNT_CREDIT_PROLONG"].notnull())]["CNT_CREDIT_PROLONG"].value_counts().sum()

plt.figure(figsize=(10,8))
plt.subplot(121)
sns.heatmap(pd.DataFrame(d),annot=True,linecolor="k",linewidths=2,cmap=sns.color_palette("Set1"))
plt.title("%age of number of times prolonged - defaulters")

plt.subplot(122)
sns.heatmap(pd.DataFrame(r),annot=True,linecolor="k",linewidths=2,cmap=sns.color_palette("Set2"))
plt.title("%ge of number of times prolonged - repayers")
plt.subplots_adjust(wspace = .7)
app_bureau_balance = app_bureau.merge(bureau_balance,left_on="SK_ID_BUREAU",right_on="SK_ID_BUREAU",how="left")
plt.figure(figsize=(13,7))
sns.boxplot(y=bureau_balance["MONTHS_BALANCE"],x=bureau_balance["STATUS"],palette="husl")
plt.title("Months balance for status types")
plt.show()
plt.figure(figsize=(12,6))
sns.distplot(bureau_balance["MONTHS_BALANCE"],color="b")
plt.title("Distribution of Month of balances")
plt.show()
x = previous_application.groupby("SK_ID_CURR")["SK_ID_PREV"].count().reset_index()
plt.figure(figsize=(13,7))
ax = sns.distplot(x["SK_ID_PREV"],color="orange")
plt.axvline(x["SK_ID_PREV"].mean(),linestyle="dashed",color="r",label="average")
plt.axvline(x["SK_ID_PREV"].std(),linestyle="dashed",color="b",label="standard deviation")
plt.axvline(x["SK_ID_PREV"].max(),linestyle="dashed",color="g",label="maximum")
plt.legend(loc="best")
plt.title("Current loan id having previous loan applications")
ax.set_facecolor("k")
cnts = previous_application["NAME_CONTRACT_TYPE"].value_counts()
import squarify
plt.figure(figsize=(8,6))
squarify.plot(cnts.values,label=cnts.keys(),value=cnts.values,linewidth=2,edgecolor="k",alpha=.8,color=sns.color_palette("Set1"))
plt.axis("off")
plt.title("Contaract types in previous applications")
plt.show()
plt.figure(figsize=(12,13))
plt.subplot(211)
ax = sns.kdeplot(previous_application["AMT_APPLICATION"],color="b",linewidth=3)
ax = sns.kdeplot(previous_application[previous_application["AMT_CREDIT"].notnull()]["AMT_CREDIT"],color="r",linewidth=3)
plt.axvline(previous_application[previous_application["AMT_CREDIT"].notnull()]["AMT_CREDIT"].mean(),color="r",linestyle="dashed",label="AMT_APPLICATION_MEAN")
plt.axvline(previous_application["AMT_APPLICATION"].mean(),color="b",linestyle="dashed",label="AMT_APPLICATION_MEAN")
plt.legend(loc="best")
plt.title("Previous loan amounts applied and loan amounts credited.")
ax.set_facecolor("k")

plt.subplot(212)
diff = (previous_application["AMT_CREDIT"] - previous_application["AMT_APPLICATION"]).reset_index()
diff = diff[diff[0].notnull()]
ax1 = sns.kdeplot(diff[0],color="g",linewidth=3,label = "difference in amount requested by client and amount credited")
plt.axvline(diff[0].mean(),color="white",linestyle="dashed",label = "mean")
plt.title("difference in amount requested by client and amount credited")
ax1.legend(loc="best")
ax1.set_facecolor("k")
mn = previous_application.groupby("NAME_CONTRACT_TYPE")[["AMT_APPLICATION","AMT_CREDIT"]].mean().stack().reset_index()
tt = previous_application.groupby("NAME_CONTRACT_TYPE")[["AMT_APPLICATION","AMT_CREDIT"]].sum().stack().reset_index()
fig = plt.figure(figsize=(10,13))
fig.set_facecolor("ghostwhite")
plt.subplot(211)
ax = sns.barplot(0,"NAME_CONTRACT_TYPE",data=mn[:6],hue="level_1",palette="inferno")
ax.set_facecolor("k")
ax.set_xlabel("average amounts")
ax.set_title("Average amounts by contract types")

plt.subplot(212)
ax1 = sns.barplot(0,"NAME_CONTRACT_TYPE",data=tt[:6],hue="level_1",palette="magma")
ax1.set_facecolor("k")
ax1.set_xlabel("total amounts")
ax1.set_title("total amounts by contract types")
plt.subplots_adjust(hspace = .2)
plt.figure(figsize=(14,5))
plt.subplot(121)
previous_application.groupby("NAME_CONTRACT_TYPE")["AMT_ANNUITY"].sum().plot(kind="bar")
plt.xticks(rotation=0)
plt.title("Total annuity amount by contract types in previous applications")
plt.subplot(122)
previous_application.groupby("NAME_CONTRACT_TYPE")["AMT_ANNUITY"].mean().plot(kind="bar")
plt.title("average annuity amount by contract types in previous applications")
plt.xticks(rotation=0)
plt.show()
ax = pd.crosstab(previous_application["NAME_CONTRACT_TYPE"],previous_application["NAME_CONTRACT_STATUS"]).plot(kind="barh",figsize=(10,7),stacked=True)
plt.xticks(rotation =0)
plt.ylabel("count")
plt.title("Count of application status by application type")
ax.set_facecolor("k")
ax = pd.crosstab(previous_application["WEEKDAY_APPR_PROCESS_START"],previous_application["NAME_CONTRACT_STATUS"]).plot(kind="barh",colors=["g","r","b","orange"],
                                                                                                                  stacked =True,figsize=(12,8))
ax.set_facecolor("k")

ax.set_title("Contract status by weekdays")
plt.show()
hr = pd.crosstab(previous_application["HOUR_APPR_PROCESS_START"],previous_application["NAME_CONTRACT_STATUS"]).stack().reset_index()
plt.figure(figsize=(12,8))
ax = sns.pointplot(hr["HOUR_APPR_PROCESS_START"],hr[0],hue=hr["NAME_CONTRACT_STATUS"],palette=["g","r","b","orange"],scale=1)
ax.set_facecolor("k")
ax.set_ylabel("count")
ax.set_title("Contract status by day hours.")
plt.grid(True,alpha=.2)
ax = pd.crosstab(previous_application["HOUR_APPR_PROCESS_START"],previous_application["WEEKDAY_APPR_PROCESS_START"]).plot(kind="bar",colors=sns.color_palette("rainbow",7),
                                                                                                                     figsize=(13,8),stacked=True)
ax.set_facecolor("k")
ax.set_title("Peak hours for week days ")
plt.show()
plt.figure(figsize=(13,6))
plt.subplot(121)
sns.violinplot(previous_application["RATE_INTEREST_PRIMARY"],alpha=.01,color="orange")
plt.axvline(previous_application[previous_application["RATE_INTEREST_PRIMARY"].notnull()]["RATE_INTEREST_PRIMARY"].mean(),color="k",linestyle="dashed")
plt.title("RATE_INTEREST_PRIMARY")
plt.subplot(122)
sns.violinplot(previous_application["RATE_INTEREST_PRIVILEGED"],color="c")
plt.axvline(previous_application[previous_application["RATE_INTEREST_PRIVILEGED"].notnull()]["RATE_INTEREST_PRIVILEGED"].mean(),color="k",linestyle="dashed")
plt.title("RATE_INTEREST_PRIVILEGED")
plt.show()
previous_application[["NAME_CASH_LOAN_PURPOSE","NAME_CONTRACT_STATUS"]]
purpose = pd.crosstab(previous_application["NAME_CASH_LOAN_PURPOSE"],previous_application["NAME_CONTRACT_STATUS"])
purpose["a"] = (purpose["Approved"]*100)/(purpose["Approved"]+purpose["Canceled"]+purpose["Refused"]+purpose["Unused offer"])
purpose["c"] = (purpose["Canceled"]*100)/(purpose["Approved"]+purpose["Canceled"]+purpose["Refused"]+purpose["Unused offer"])
purpose["r"] = (purpose["Refused"]*100)/(purpose["Approved"]+purpose["Canceled"]+purpose["Refused"]+purpose["Unused offer"])
purpose["u"] = (purpose["Unused offer"]*100)/(purpose["Approved"]+purpose["Canceled"]+purpose["Refused"]+purpose["Unused offer"])
purpose_new = purpose[["a","c","r","u"]]
purpose_new = purpose_new.stack().reset_index()
purpose_new["NAME_CONTRACT_STATUS"] = purpose_new["NAME_CONTRACT_STATUS"].replace({"a":"accepted_percentage","c":"cancelled_percentage",
                                                               "r":"refused_percentage","u":"unused_percentage"})

lst = purpose_new["NAME_CONTRACT_STATUS"].unique().tolist()
length = len(lst)
cs = ["lime","orange","r","b"]

fig = plt.figure(figsize=(14,18))
fig.set_facecolor("lightgrey")
for i,j,k in itertools.zip_longest(lst,range(length),cs):
    plt.subplot(2,2,j+1)
    dat = purpose_new[purpose_new["NAME_CONTRACT_STATUS"] == i]
    ax = sns.barplot(0,"NAME_CASH_LOAN_PURPOSE",data=dat.sort_values(by=0,ascending=False),color=k)
    plt.ylabel("")
    plt.xlabel("percentage")
    plt.title(i+" by purpose")
    plt.subplots_adjust(wspace = .7)
    ax.set_facecolor("k")

plt.figure(figsize=(13,6))
sns.violinplot(y= previous_application["DAYS_DECISION"],
               x = previous_application["NAME_CONTRACT_STATUS"],palette=["r","g","b","y"])
plt.axhline(previous_application[previous_application["NAME_CONTRACT_STATUS"] == "Approved"]["DAYS_DECISION"].mean(),
            color="r",linestyle="dashed",label="accepted_average")
plt.axhline(previous_application[previous_application["NAME_CONTRACT_STATUS"] == "Refused"]["DAYS_DECISION"].mean(),
            color="g",linestyle="dashed",label="refused_average")
plt.axhline(previous_application[previous_application["NAME_CONTRACT_STATUS"] == "Cancelled"]["DAYS_DECISION"].mean(),color="b",
            linestyle="dashed",label="cancelled_average")
plt.axhline(previous_application[previous_application["NAME_CONTRACT_STATUS"] == "Unused offer"]["DAYS_DECISION"].mean(),color="y",
            linestyle="dashed",label="un used_average")
plt.legend(loc="best")

plt.title("Contract status relative to decision made about previous application.")
plt.show()
plt.figure(figsize=(8,12))
plt.subplot(211)
rej = previous_application["CODE_REJECT_REASON"].value_counts().reset_index()
ax = sns.barplot("CODE_REJECT_REASON","index",data=rej[:6],palette="husl")
for i,j in enumerate(np.around((rej["CODE_REJECT_REASON"][:6].values*100/(rej["CODE_REJECT_REASON"][:6].sum())))):
    ax.text(.7,i,j,weight="bold")
plt.xlabel("percentage")
plt.ylabel("CODE_REJECT_REASON")
plt.title("Reasons for application rejections")

plt.subplot(212)
pay = previous_application["NAME_PAYMENT_TYPE"].value_counts().reset_index()
ax1 = sns.barplot("NAME_PAYMENT_TYPE","index",data=pay,palette="husl")
for i,j in enumerate(np.around((pay["NAME_PAYMENT_TYPE"].values*100/(pay["NAME_PAYMENT_TYPE"].sum())))):
    ax1.text(.7,i,j,weight="bold")
plt.xlabel("percentage")
plt.ylabel("NAME_PAYMENT_TYPE")
plt.title("Clients payment methods")
plt.subplots_adjust(hspace = .3)
plt.figure(figsize=(13,6))
plt.subplot(121)
previous_application["NAME_TYPE_SUITE"].value_counts().plot.pie(autopct = "%1.1f%%",fontsize=8,
                                                             colors = sns.color_palette("inferno"),
                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)
circ = plt.Circle((0,0),.7,color="white")
plt.gca().add_artist(circ)
plt.title("NAME_TYPE_SUITE")

plt.subplot(122)
previous_application["NAME_CLIENT_TYPE"].value_counts().plot.pie(autopct = "%1.1f%%",fontsize=8,
                                                             colors = sns.color_palette("inferno"),
                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)
circ = plt.Circle((0,0),.7,color="white")
plt.gca().add_artist(circ)
plt.title("NAME_CLIENT_TYPE")
plt.show()
goods = previous_application["NAME_GOODS_CATEGORY"].value_counts().reset_index()
goods["percentage"] = round(goods["NAME_GOODS_CATEGORY"]*100/goods["NAME_GOODS_CATEGORY"].sum(),2)
fig = plt.figure(figsize=(12,5))
ax = sns.pointplot("index","percentage",data=goods,color="yellow")
plt.xticks(rotation = 80)
plt.xlabel("NAME_GOODS_CATEGORY")
plt.ylabel("percentage")
plt.title("popular goods for applying loans")
ax.set_facecolor("k")
fig.set_facecolor('lightgrey')
plt.figure(figsize=(12,6))
plt.subplot(121)
previous_application["NAME_PORTFOLIO"].value_counts().plot.pie(autopct = "%1.1f%%",fontsize=8,
                                                             colors = sns.color_palette("prism",5),
                                              wedgeprops={"linewidth":2,"edgecolor":"white"},
                                                               shadow =True)
plt.title("previous applications portfolio")
plt.subplot(122)
previous_application["NAME_PRODUCT_TYPE"].value_counts().plot.pie(autopct = "%1.1f%%",fontsize=8,
                                                             colors = sns.color_palette("prism",3),
                                              wedgeprops={"linewidth":2,"edgecolor":"white"},
                                                                  shadow =True)
plt.title("previous applications product types")
plt.show()
app = pd.crosstab(previous_application["CHANNEL_TYPE"],previous_application["NAME_CONTRACT_STATUS"])
app1 = app
app1["approval_rate"] = app1["Approved"]*100/(app1["Approved"]+app1["Refused"]+app1["Canceled"])
app1["refused_rate"]  = app1["Refused"]*100/(app1["Approved"]+app1["Refused"]+app1["Canceled"])
app1["cacelled_rate"] = app1["Canceled"]*100/(app1["Approved"]+app1["Refused"]+app1["Canceled"])
app2 = app[["approval_rate","refused_rate","cacelled_rate"]]
ax = app2.plot(kind="barh",stacked=True,figsize=(10,7))
ax.set_facecolor("k")
ax.set_xlabel("percentage")
ax.set_title("approval,cancel and refusal rates by channel types")
plt.show()
fig = plt.figure(figsize=(13,5))
plt.subplot(121)
are = previous_application.groupby("SELLERPLACE_AREA")["AMT_CREDIT"].sum().reset_index()
are = are.sort_values(by ="AMT_CREDIT",ascending = False)
ax = sns.barplot(y= "AMT_CREDIT",x ="SELLERPLACE_AREA",data=are[:15],color="r")
ax.set_facecolor("k")
ax.set_title("Highest amount credited seller place areas")

plt.subplot(122)
sell = previous_application.groupby("NAME_SELLER_INDUSTRY")["AMT_CREDIT"].sum().reset_index().sort_values(by = "AMT_CREDIT",ascending = False)
ax1=sns.barplot(y = "AMT_CREDIT",x = "NAME_SELLER_INDUSTRY",data=sell,color="b")
ax1.set_facecolor("k")
ax1.set_title("Highest amount credited seller industrys")
plt.xticks(rotation=90)
plt.subplots_adjust(wspace = .5)
fig.set_facecolor("lightgrey")
plt.figure(figsize=(13,5))
ax = sns.countplot(previous_application["CNT_PAYMENT"],palette="Set1",order=previous_application["CNT_PAYMENT"].value_counts().index)
ax.set_facecolor("k")
plt.xticks(rotation = 90)
plt.title("popular terms of previous credit at application")
plt.show()
plt.figure(figsize=(10,8))
sns.countplot(y = previous_application["PRODUCT_COMBINATION"],order=previous_application["PRODUCT_COMBINATION"].value_counts().index)
plt.title("Detailed product combination of the previous application -count")
plt.show()
plt.figure(figsize=(12,6))
plt.subplot(121)
previous_application["NFLAG_INSURED_ON_APPROVAL"].value_counts().plot.pie(autopct = "%1.1f%%",fontsize=8,
                                                             colors = sns.color_palette("prism",4),
                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)
circ = plt.Circle((0,0),.7,color="white")
plt.gca().add_artist(circ)
plt.title("client requesting insurance")

plt.subplot(122)
previous_application["NAME_YIELD_GROUP"].value_counts().plot.pie(autopct = "%1.1f%%",fontsize=8,
                                                             colors = sns.color_palette("prism",4),
                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)
circ = plt.Circle((0,0),.7,color="white")
plt.gca().add_artist(circ)
plt.title("interest rates")
plt.show()
cols = ['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION','DAYS_LAST_DUE', 'DAYS_TERMINATION']
plt.figure(figsize=(12,6))
sns.heatmap(previous_application[cols].describe()[1:].transpose(),
            annot=True,linewidth=2,linecolor="k",cmap=sns.color_palette("inferno"))
plt.show()
cols = ['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION','DAYS_LAST_DUE', 'DAYS_TERMINATION']
length = len(cols)
cs = ["r","g","b","c","m"]

plt.figure(figsize=(13,6))
for i,j,k in itertools.zip_longest(cols,range(length),cs):
    ax = sns.distplot(previous_application[previous_application[i].notnull()][i],color=k,label=i)
    plt.legend(loc="best")
    ax.set_facecolor("k")
    plt.xlabel("days")
    plt.title("Days variables - Relative to application date of current application")
plt.figure(figsize=(14,6))
ax = sns.countplot(pos_cash_balance["MONTHS_BALANCE"],palette="rainbow")
plt.xticks(rotation = 90,fontsize=8)
plt.title("frequency distribution in Month of balance relative to application date for previous applications")
ax.set_facecolor("k")
plt.show()
fig = plt.figure(figsize=(12,7))
plt.subplot(122)
sns.countplot(pos_cash_balance["CNT_INSTALMENT"],
              order=pos_cash_balance["CNT_INSTALMENT"].value_counts().index[:10],palette="husl")
plt.title("Term of previous credit")
plt.subplot(121)
sns.countplot(pos_cash_balance["CNT_INSTALMENT_FUTURE"],
              order=pos_cash_balance["CNT_INSTALMENT_FUTURE"].value_counts().index[:10],palette="husl")
plt.title("Installments left to pay on the previous credit.")
fig.set_facecolor("lightgrey")
amt_ins = installments_payments.groupby("SK_ID_CURR").agg({"AMT_INSTALMENT":"sum","AMT_PAYMENT":"sum","SK_ID_PREV":"count"}).reset_index()

plt.figure(figsize = (12,8))
plt.scatter(amt_ins["AMT_INSTALMENT"],amt_ins["AMT_PAYMENT"],
            c=amt_ins["SK_ID_PREV"],edgecolor="k",cmap="viridis",s=60)
plt.colorbar()
plt.xlabel("AMT_INSTALMENT")
plt.ylabel("AMT_PAYMENT")
plt.title("Total prescribed installment amount and total amount actually paid to client on previous credits.")
plt.show()
days_ins = installments_payments.groupby("SK_ID_CURR")[["DAYS_ENTRY_PAYMENT","DAYS_INSTALMENT"]].mean().reset_index()
days_ins = days_ins[days_ins["DAYS_ENTRY_PAYMENT"].notnull()]
plt.figure(figsize=(12,9))
plt.hist2d(days_ins["DAYS_ENTRY_PAYMENT"],days_ins["DAYS_INSTALMENT"],bins=(20,20),cmap="hot")
plt.colorbar()
plt.xlabel("AVERAGE_DAYS_ENTRY_PAYMENT")
plt.ylabel("AVERAGE_DAYS_INSTALMENT")
plt.title("density plot between Average days previous credit was supposed to be paid and credit actually paid by current ids.")
plt.show()
cols = ['AMT_BALANCE','AMT_CREDIT_LIMIT_ACTUAL', 'AMT_INST_MIN_REGULARITY',
       'AMT_RECEIVABLE_PRINCIPAL', 'AMT_RECIVABLE', 'AMT_TOTAL_RECEIVABLE']

length = len(cols)
cs = ["r","g","b","c","m","y"]

fig = plt.figure(figsize=(13,14))
fig.set_facecolor("lightgrey")

for i,j,k in itertools.zip_longest(cols,range(length),cs):
    plt.subplot(3,2,j+1)
    ax = sns.distplot(credit_card_balance[credit_card_balance[i].notnull()][i],color=k)
    ax.set_facecolor("k")
    plt.xlabel("")
    plt.title(i)
application_train["type"] = "train"
application_test["type"]  = "test"
#conactenating train & test data
data = pd.concat([application_train,application_test],axis=0)
#Removing columns with missing values more than 40%
missing_cols = [ 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
       'APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG',
       'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG',
       'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG',
       'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG',
       'NONLIVINGAREA_AVG', 'APARTMENTS_MODE', 'BASEMENTAREA_MODE',
       'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BUILD_MODE', 'COMMONAREA_MODE',
       'ELEVATORS_MODE', 'ENTRANCES_MODE', 'FLOORSMAX_MODE', 'FLOORSMIN_MODE',
       'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE',
       'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE', 'APARTMENTS_MEDI',
       'BASEMENTAREA_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI',
       'COMMONAREA_MEDI', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI', 'FLOORSMAX_MEDI',
       'FLOORSMIN_MEDI', 'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI',
       'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI',
       'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'TOTALAREA_MODE',
       'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE',"OWN_CAR_AGE","OCCUPATION_TYPE"]

data_new  = data[[i for i in data.columns if i not in missing_cols]]
#Separating numberical and categorical columns
obj_dtypes = [i for i in data_new.select_dtypes(include=np.object).columns if i not in ["type"] ]
num_dtypes = [i for i in data_new.select_dtypes(include = np.number).columns if i not in ['SK_ID_CURR'] + [ 'TARGET']]
#MISSING values treatment
amt_cs = ["AMT_ANNUITY","AMT_GOODS_PRICE"]
for i in amt_cs:
    data_new[i] = data_new.groupby("type").transform(lambda x:x.fillna(x.mean()))
    
enq_cs =['AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_HOUR',
       'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT',
       'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_YEAR']
for i in enq_cs:
    data_new[i] = data_new[i].fillna(0)
    
cols = ["DEF_30_CNT_SOCIAL_CIRCLE","DEF_60_CNT_SOCIAL_CIRCLE","OBS_30_CNT_SOCIAL_CIRCLE",
        "OBS_60_CNT_SOCIAL_CIRCLE","NAME_TYPE_SUITE","CNT_FAM_MEMBERS",
       "DAYS_LAST_PHONE_CHANGE","DAYS_LAST_PHONE_CHANGE"]
for i in cols :
    data_new[i]  = data_new[i].fillna(data_new[i].mode()[0])
#Label encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for i in obj_dtypes:
    data_new[i] = le.fit_transform(data_new[i])
#one hot encoding for categorical variables
data_new = pd.get_dummies(data=data_new,columns=obj_dtypes)
#splitting new train and test data
application_train_newdf = data_new[data_new["type"] == "train"]
application_test_newdf  = data_new[data_new["type"] == "test"]
#splitting application_train_newdf into train and test
from sklearn.model_selection import train_test_split
train,test = train_test_split(application_train_newdf,test_size=.3,random_state = 123)

train = train.drop(columns="type",axis=1)
test  = test.drop(columns="type",axis=1)

#seperating dependent and independent variables
train_X = train[[i for i in train.columns if i not in ['SK_ID_CURR'] + [ 'TARGET']]]
train_Y = train[["TARGET"]]

test_X  = test[[i for i in test.columns if i not in ['SK_ID_CURR'] + [ 'TARGET']]]
test_Y  = test[["TARGET"]]
# Up-sample Minority Class
from sklearn.utils import resample

#separating majority and minority classes
df_majority = train[train["TARGET"] == 0]
df_minority = train[train["TARGET"] == 1]

#upsample minority data
df_minority_upsampled = resample(df_minority,
                                 replace=True,
                                 n_samples =197969,
                                 random_state=123)

df_upsampled = pd.concat([df_majority,df_minority_upsampled],axis=0)

#splitting dependent and independent variables
df_upsampled_X = df_upsampled[[i for i in df_upsampled.columns if i not in ['SK_ID_CURR'] + [ 'TARGET']]]
df_upsampled_Y = df_upsampled[["TARGET"]]
# Down-sample Majority Class
from sklearn.utils import resample

#separating majority and minority classes
df_majority = train[train["TARGET"] == 0]
df_minority = train[train["TARGET"] == 1]

df_majority_downsampled = resample(df_majority,
                                   replace=False,
                                   n_samples=17288,
                                   random_state=123)

df_downsampled = pd.concat([df_minority,df_majority_downsampled],axis=0)

#splitting dependent and independent variables

df_downsampled_X = df_downsampled[[i for i in df_downsampled.columns if i not in ['SK_ID_CURR'] + [ 'TARGET']]]
df_downsampled_Y = df_downsampled[["TARGET"]]
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,roc_auc_score,classification_report,roc_auc_score,roc_curve,auc

#Model function
def model(algorithm,dtrain_X,dtrain_Y,dtest_X,dtest_Y,cols=None):

    algorithm.fit(dtrain_X[cols],dtrain_Y)
    predictions = algorithm.predict(dtest_X[cols])
    print (algorithm)
    
    print ("Accuracy score : ", accuracy_score(predictions,dtest_Y))
    print ("Recall score   : ", recall_score(predictions,dtest_Y))
    print ("classification report :\n",classification_report(predictions,dtest_Y))
    
    fig = plt.figure(figsize=(10,8))
    ax  = fig.add_subplot(111)
    prediction_probabilities = algorithm.predict_proba(dtest_X[cols])[:,1]
    fpr , tpr , thresholds   = roc_curve(dtest_Y,prediction_probabilities)
    ax.plot(fpr,tpr,label   = ["Area under curve : ",auc(fpr,tpr)],linewidth=2,linestyle="dotted")
    ax.plot([0,1],[0,1],linewidth=2,linestyle="dashed")
    plt.legend(loc="best")
    plt.title("ROC-CURVE & AREA UNDER CURVE")
    ax.set_facecolor("k")

from sklearn.linear_model import LogisticRegression
logit = LogisticRegression()
model(logit,df_downsampled_X,df_downsampled_Y,test_X,test_Y,df_downsampled_X.columns)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
model(rfc,df_downsampled_X,df_downsampled_Y,test_X,test_Y,df_downsampled_X.columns)
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
model(dtc,df_downsampled_X,df_downsampled_Y,test_X,test_Y,df_downsampled_X.columns)
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
model(gnb,df_downsampled_X,df_downsampled_Y,test_X,test_Y,df_downsampled_X.columns)
from xgboost import XGBClassifier
xgb = XGBClassifier()
model(xgb,df_downsampled_X,df_downsampled_Y,test_X,test_Y,df_downsampled_X.columns)
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
model(gbc,df_downsampled_X,df_downsampled_Y,test_X,test_Y,df_downsampled_X.columns)
test_sub_X = application_test_newdf[[i for i in df_downsampled.columns if i not in ['SK_ID_CURR'] + [ 'TARGET']]]
xgb1 = XGBClassifier()
xgb1.fit(df_downsampled_X,df_downsampled_Y)
sub_prob = xgb1.predict_proba(test_sub_X)[:,1]
sub_prob = pd.DataFrame(sub_prob)
ids = application_test[["SK_ID_CURR"]]
subm  = ids.merge(sub_prob,left_index=True,right_index=True,how="left")
sample_submission  = subm.rename(columns={"SK_ID_CURR":'SK_ID_CURR',0:'TARGET'})
