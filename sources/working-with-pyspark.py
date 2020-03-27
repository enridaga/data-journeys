
!pip install pyspark
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import *
spark = SparkSession.builder.appName('data_processing').getOrCreate()
from pyspark.sql.functions import udf
df = spark.read.csv("../input/sales/Sales Records.csv",header=True,inferSchema=True)
df.show(5)
df.dtypes
df = df.withColumnRenamed('Item Type','Item_Type').withColumnRenamed('Sales Channel','Sales_Channel').withColumnRenamed('Order Priority','Order_Priority')
df = df.withColumnRenamed('Order Date','Order_Date').withColumnRenamed('Order ID','Order_ID').withColumnRenamed('Ship Date','Ship_Date')
df = df.withColumnRenamed('Units Sold','Units_Sold').withColumnRenamed('Unit Price','Unit_Price').withColumnRenamed('Unit Cost','Unit_Cost')
df = df.withColumnRenamed('Total Revenue','Total_Revenue').withColumnRenamed('Total Cost','Total_Cost').withColumnRenamed('Total Profit','Total_Profit')
df.select(['Item_Type']).distinct().show()
df.select(['Region','Country']).distinct().show()
df[(df.Order_Priority == 'H') 
   & (df.Country == 'United States of America') ].show(5,False)
df.filter(df.Sales_Channel == 'Online').filter(df.Region == 'North America').show(5,False)
df.filter(df.Sales_Channel == 'Offline').filter(df.Region == 'North America').show(5,False)
df.where((df.Units_Sold > 5000) | (df.Total_Revenue >= 100000)).show(5,False)
df.where((df.Order_Priority == 'H') & (df.Total_Profit >= 1000000)).show(5,False)
df.groupby('Region').agg(F.sum('Total_Profit').alias('Region Profits')).show(5,False)
df.groupby('Item_Type').agg(F.sum('Total_Profit').alias('Item Profits')).show(5,False)
df.groupby('Country').agg(F.sum('Total_Profit').alias('Sum Total Profits')).show(5,False)
df.groupby('Sales_Channel').agg(F.sum('Total_Profit').alias('Channel Profits')).show(5,False)
df.groupby('Region').agg(F.mean('Total_Revenue').alias('Sum Total Revenue')).show(5,False)
df.groupby('Country').agg(F.mean('Total_Revenue').alias('Average Revenue')).show(5,False)
df.groupby('Region').agg(F.mean('Total_Cost').alias('Item Cost')).show(5,False)
df.groupby('Region').agg(F.collect_set('Item_Type')).show(5)
df.groupby("Region").agg(F.collect_list("Units_Sold")).show()
def Prof(Total_Profit):
    if Total_Profit >= 937196.46:
        return 'Above Average Profit'
    else: 
        return 'Below Average Profit'
Prof_udf=udf(Prof,StringType())
df=df.withColumn('Prof',Prof_udf(df['Total_Profit']))
df.select('Total_Profit','Prof').show(5,False)
df[df.Total_Profit >= 950000].show(5)
df1 = spark.read.csv('../input/working13/ManagerInformation.csv',header=True,inferSchema=True)
df1.show()
join_df = df.join(df1,on='Region')
join_df.groupby('Manager_Name').agg(F.sum('Total_Profit').alias('Manager Profits')).show(5,False)
df.groupby('Region').pivot('Item_Type').sum('Total_Revenue').fillna(0).show(5,False)
from pyspark.sql.window import Window
from pyspark.sql.functions import col,row_number
WinF = Window.orderBy(df['Total_Profit'].desc())
df = df.withColumn('rank',row_number().over(WinF).alias('rank'))
df.show(5)