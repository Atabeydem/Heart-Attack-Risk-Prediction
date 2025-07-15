import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


df = pd.read_csv('heart.csv')

#print(df.head())

#print(df.info)
#print(df.isnull().sum())

categorical_columns = ['ChestPainType', 'Sex','RestingBP','ExerciseAngina','ST_Slope']

le = LabelEncoder()
for column in categorical_columns:
    df[column] = le.fit_transform(df[column])


numeric_columns= ['Age','RestingBP','Cholesterol','FastingBS','MaxHR','Oldpeak']

scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
print(df.head())

df.to_csv('heart_edited.csv', index =False)
