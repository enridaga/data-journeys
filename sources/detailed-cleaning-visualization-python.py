
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

### pylab inline

pylab.rcParams['figure.figsize'] = (10, 6)
limit_rows   = 7000000

df           = pd.read_csv("../input/train_ver2.csv",dtype={"sexo":str,

                                                    "ind_nuevo":str,

                                                    "ult_fec_cli_1t":str,

                                                    "indext":str}, nrows=limit_rows)

unique_ids   = pd.Series(df["ncodpers"].unique())

limit_people = 1.2e4

unique_id    = unique_ids.sample(n=limit_people)

df           = df[df.ncodpers.isin(unique_id)]

df.describe()
df["fecha_dato"] = pd.to_datetime(df["fecha_dato"],format="%Y-%m-%d")

df["fecha_alta"] = pd.to_datetime(df["fecha_alta"],format="%Y-%m-%d")

df["fecha_dato"].unique()
df["month"] = pd.DatetimeIndex(df["fecha_dato"]).month

df["age"]   = pd.to_numeric(df["age"], errors="coerce")
df.isnull().any()
with sns.plotting_context("notebook",font_scale=1.5):

    sns.set_style("whitegrid")

    sns.distplot(df["age"].dropna(),

                 bins=80,

                 kde=False,

                 color="tomato")

    sns.plt.title("Age Distribution")

    plt.ylabel("Count")
df.loc[df.age < 18,"age"]  = df.loc[(df.age >= 18) & (df.age <= 30),"age"].mean(skipna=True)

df.loc[df.age > 100,"age"] = df.loc[(df.age >= 30) & (df.age <= 100),"age"].mean(skipna=True)

df["age"].fillna(df["age"].mean(),inplace=True)

df["age"]                  = df["age"].astype(int)
with sns.plotting_context("notebook",font_scale=1.5):

    sns.set_style("whitegrid")

    sns.distplot(df["age"].dropna(),

                 bins=80,

                 kde=False,

                 color="tomato")

    sns.plt.title("Age Distribution")

    plt.ylabel("Count")

    plt.xlim((15,100))
df["ind_nuevo"].isnull().sum()
months_active = df.loc[df["ind_nuevo"].isnull(),:].groupby("ncodpers", sort=False).size()

months_active.max()
df.loc[df["ind_nuevo"].isnull(),"ind_nuevo"] = 1
df.antiguedad = pd.to_numeric(df.antiguedad,errors="coerce")

np.sum(df["antiguedad"].isnull())
df.loc[df["antiguedad"].isnull(),"ind_nuevo"].describe()
df.loc[df.antiguedad.isnull(),"antiguedad"] = df.antiguedad.min()

df.loc[df.antiguedad <0, "antiguedad"]      = 0 # Thanks @StephenSmith for bug-find
dates=df.loc[:,"fecha_alta"].sort_values().reset_index()

median_date = int(np.median(dates.index.values))

df.loc[df.fecha_alta.isnull(),"fecha_alta"] = dates.loc[median_date,"fecha_alta"]

df["fecha_alta"].describe()
pd.Series([i for i in df.indrel]).value_counts()
df.loc[df.indrel.isnull(),"indrel"] = 1
df.drop(["tipodom","cod_prov"],axis=1,inplace=True)
df.isnull().any()
np.sum(df["ind_actividad_cliente"].isnull())
df.loc[df.ind_actividad_cliente.isnull(),"ind_actividad_cliente"] = \

df["ind_actividad_cliente"].median()
df.nomprov.unique()
df.loc[df.nomprov=="CORU\xc3\x91A, A","nomprov"] = "CORUNA, A"
df.loc[df.nomprov.isnull(),"nomprov"] = "UNKNOWN"
df.renta.isnull().sum()
#df.loc[df.renta.notnull(),:].groupby("nomprov").agg([{"Sum":sum},{"Mean":mean}])

incomes = df.loc[df.renta.notnull(),:].groupby("nomprov").agg({"renta":{"MedianIncome":median}})

incomes.sort_values(by=("renta","MedianIncome"),inplace=True)

incomes.reset_index(inplace=True)

incomes.nomprov = incomes.nomprov.astype("category", categories=[i for i in df.nomprov.unique()],ordered=False)

incomes.head()
with sns.axes_style({

        "axes.facecolor":   "#ffc400",

        "axes.grid"     :    False,

        "figure.facecolor": "#c60b1e"}):

    h = sns.factorplot(data=incomes,

                   x="nomprov",

                   y=("renta","MedianIncome"),

                   order=(i for i in incomes.nomprov),

                   size=6,

                   aspect=1.5,

                   scale=1.0,

                   color="#c60b1e",

                   linestyles="None")

plt.xticks(rotation=90)

plt.tick_params(labelsize=16,labelcolor="#ffc400")#

plt.ylabel("Median Income",size=32,color="#ffc400")

plt.xlabel("City",size=32,color="#ffc400")

plt.title("Income Distribution by City",size=40,color="#ffc400")

plt.ylim(0,180000)

plt.yticks(range(0,180000,40000))
grouped        = df.groupby("nomprov").agg({"renta":lambda x: x.median(skipna=True)}).reset_index()

new_incomes    = pd.merge(df,grouped,how="inner",on="nomprov").loc[:, ["nomprov","renta_y"]]

new_incomes    = new_incomes.rename(columns={"renta_y":"renta"}).sort_values("renta").sort_values("nomprov")

df.sort_values("nomprov",inplace=True)

df             = df.reset_index()

new_incomes    = new_incomes.reset_index()
df.loc[df.renta.isnull(),"renta"] = new_incomes.loc[df.renta.isnull(),"renta"].reset_index()

df.loc[df.renta.isnull(),"renta"] = df.loc[df.renta.notnull(),"renta"].median()

df.sort_values(by="fecha_dato",inplace=True)
df.ind_nomina_ult1.isnull().sum()
df.loc[df.ind_nomina_ult1.isnull(), "ind_nomina_ult1"] = 0

df.loc[df.ind_nom_pens_ult1.isnull(), "ind_nom_pens_ult1"] = 0
string_data = df.select_dtypes(include=["object"])

missing_columns = [col for col in string_data if string_data[col].isnull().any()]

for col in missing_columns:

    print("Unique values for {0}:\n{1}\n".format(col,string_data[col].unique()))

del string_data
df.loc[df.indfall.isnull(),"indfall"] = "N"

df.loc[df.tiprel_1mes.isnull(),"tiprel_1mes"] = "A"

df.tiprel_1mes = df.tiprel_1mes.astype("category")



# As suggested by @StephenSmith

map_dict = { 1.0  : "1",

            "1.0" : "1",

            "1"   : "1",

            "3.0" : "3",

            "P"   : "P",

            3.0   : "3",

            2.0   : "2",

            "3"   : "3",

            "2.0" : "2",

            "4.0" : "4",

            "4"   : "4",

            "2"   : "2"}



df.indrel_1mes.fillna("P",inplace=True)

df.indrel_1mes = df.indrel_1mes.apply(lambda x: map_dict.get(x,x))

df.indrel_1mes = df.indrel_1mes.astype("category")





unknown_cols = [col for col in missing_columns if col not in ["indfall","tiprel_1mes","indrel_1mes"]]

for col in unknown_cols:

    df.loc[df[col].isnull(),col] = "UNKNOWN"
df.isnull().any()
feature_cols = df.iloc[:1,].filter(regex="ind_+.*ult.*").columns.values

for col in feature_cols:

    df[col] = df[col].astype(int)
unique_months = pd.DataFrame(pd.Series(df.fecha_dato.unique()).sort_values()).reset_index(drop=True)

unique_months["month_id"] = pd.Series(range(1,1+unique_months.size)) # start with month 1, not 0 to match what we already have

unique_months["month_next_id"] = 1 + unique_months["month_id"]

unique_months.rename(columns={0:"fecha_dato"},inplace=True)

df = pd.merge(df,unique_months,on="fecha_dato")
def status_change(x):

    diffs = x.diff().fillna(0)# first occurrence will be considered Maintained, 

    #which is a little lazy. A better way would be to check if 

    #the earliest date was the same as the earliest we have in the dataset

    #and consider those separately. Entries with earliest dates later than that have 

    #joined and should be labeled as "Added"

    label = ["Added" if i==1 \

         else "Dropped" if i==-1 \

         else "Maintained" for i in diffs]

    return label
# df.loc[:, feature_cols] = df..groupby("ncodpers").apply(status_change)

df.loc[:, feature_cols] = df.loc[:, [i for i in feature_cols]+["ncodpers"]].groupby("ncodpers").transform(status_change)
df = pd.melt(df, id_vars   = [col for col in df.columns if col not in feature_cols],

            value_vars= [col for col in feature_cols])

df = df.loc[df.value!="Maintained",:]

df.shape
# For thumbnail

pylab.rcParams['figure.figsize'] = (6, 4)

with sns.axes_style({

        "axes.facecolor":   "#ffc400",

        "axes.grid"     :    False,

        "figure.facecolor": "#c60b1e"}):

    h = sns.factorplot(data=incomes,

                   x="nomprov",

                   y=("renta","MedianIncome"),

                   order=(i for i in incomes.nomprov),

                   size=6,

                   aspect=1.5,

                   scale=0.75,

                   color="#c60b1e",

                   linestyles="None")

plt.xticks(rotation=90)

plt.tick_params(labelsize=12,labelcolor="#ffc400")#

plt.ylabel("Median Income",size=32,color="#ffc400")

plt.xlabel("City",size=32,color="#ffc400")

plt.title("Income Distribution by City",size=40,color="#ffc400")

plt.ylim(0,180000)

plt.yticks(range(0,180000,40000))