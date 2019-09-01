import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
data=pd.read_csv('C:\\Users\\Gowshik Saravanan\\Desktop\\7th sem\\mine.csv')
df = pd.DataFrame(data)
min_max=MinMaxScaler()
print("min-max normalisation")
print("Normalised data of age")
print(min_max.fit_transform(df[['Age']]))
print("Normalised data of Overall")
print(min_max.fit_transform(df[['Overall']]))
print("Normalised data of Potential")
print(min_max.fit_transform(df[['Potential']]))


print("Z-Score normalisation")
print("Normalised data of age")
a=np.array(df['Age'])
print(stats.zscore(a))
print("Normalised data of Overall")
b=np.array(df['Overall'])
print(stats.zscore(b))
print("Normalised data of Potential")
c=np.array(df['Potential'])
print(stats.zscore(c))
