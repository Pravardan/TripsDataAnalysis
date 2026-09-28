import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

path="UberDataset.csv"
df=pd.read_csv(path)
df.head(5)
print(df.shape) # (1156,7)
print(df.columns.tolist())

#preprocessing
df['PURPOSE'].fillna("NOT", inplace=True) 
df["starting_date"]=pd.to_datetime(df["START_DATE"],errors="coerce")
df["ending_date"]=pd.to_datetime(df['END_DATE'],errors="coerce")


df['date']=pd.DatetimeIndex(df["starting_date"]).date
df['time']=pd.DatetimeIndex(df["starting_date"]).hour

# change into day and night
df['day_night'] = pd.cut(x=df['time'],bins = [0,10,15,19,24],labels = ['Morning','Afternoon','Evening','Night']) 
df.dropna(inplace=True)

#drop duplicates from dataset
print(df.drop_duplicates(inplace=True))

#visualise the data 
obj = (df.dtypes == 'object')
object_cols=list(obj[obj].index)
unique_values= {}
for cols in object_cols:
    unique_values[cols]=df[cols].unique().size
print(unique_values)

#visualize 1
plt.figure(figsize=(10,3))
plt.subplot(1,2,1)
plt.figure(figsize=(10,5))
sns.countplot(df['CATEGORY'])
plt.xticks(rotation=90)
plt.show()
#visualize 2
plt.subplot(1,2,2)
sns.countplot(df['PURPOSE'])
plt.xticks(rotation=90)
plt.show()
#visualize 3
sns.countplot(df['day_night'])
plt.xticks(rotation=90)
plt.show()
#visualize 4
plt.figure(figsize=(10,5))
sns.countplot(data=df,x='PURPOSE',hue='CATEGORY')
plt.xticks(rotation=90)
plt.show()
#visualize 5
object_cols = ['CATEGORY', 'PURPOSE']
One_encoder = OneHotEncoder(drop='first')

# .fit_transform() returns a sparse matrix
OH_sparse_matrix = One_encoder.fit_transform(df[['PURPOSE', 'CATEGORY']])

#.toarray() to convert the sparse matrix to a dense NumPy array
OH_cols = pd.DataFrame(
    OH_sparse_matrix.toarray(), 
    columns=One_encoder.get_feature_names_out(),
    index=df.index  
)

# Remove the redundant column assignment line that threw the original error
df_final = df.drop(object_cols, axis=1)
df = pd.concat([df_final, OH_cols], axis=1)

# Generate the correlation heatmap
numeric_dataset = df.select_dtypes(include=['number'])
plt.figure(figsize=(12, 8)) # Added figure size so the heatmap isn't cramped
sns.heatmap(numeric_dataset.corr(), 
            cmap='BrBG', 
            fmt='.2f', 
            linewidths=2, 
            annot=True)
plt.show()