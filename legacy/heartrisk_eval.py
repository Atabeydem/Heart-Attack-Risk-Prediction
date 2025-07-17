import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('heart_edited.csv')

#print(df.describe())

df.hist(figsize=(18,12), bins = 20, edgecolor = 'black')
plt.suptitle('data distribution')
#plt.show()

plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='Age', y='Cholesterol', hue= 'HeartDisease')
plt.title('The Connection Between Age and Cholesterol')
#plt.show()

#label_encoder = LabelEncoder()
#df['Sex'] = label_encoder.fit_transform(df['Sex'])
#df['ChestPainType'] = label_encoder.fit_transform(df['ChestPainType'])
#df['RestingECG'] = label_encoder.fit_transform(df['RestingECG'])
#df['ExerciseAngina'] = label_encoder.fit_transform(df['ExerciseAngina'])
#df['ST_Slope'] = label_encoder.fit_transform(df['ST_Slope'])


plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot = True, cmap='coolwarm', fmt = '.2f')
plt.title('Correlation Heat Map')
plt.show()

plt.figure(figsize=(6,6))
sns.countplot(x ='HeartDisease', data=df)
plt.title('Heart Disease Prevention Rates')
plt.show()

