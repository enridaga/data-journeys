
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
df=pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')
df.head()
plt.figure(dpi=150)
plt.title('Correlation Heatmap for Student Performance')
sns.heatmap(df.corr(),annot=True,lw=1,linecolor='white',cmap='viridis')