import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

df = pd.read_csv('weatherAUS.csv')
print(df.shape)
print(df.info())
print(df.isna().sum())
#drop columns
df = df.drop(["Evaporation","Sunshine","Cloud9am","Cloud3pm","Location", "Date"], axis =1)

print(df.describe());
df = df.dropna(axis = 0);
print(df.shape)
print(df.duplicated())
df.drop_duplicates()
from sklearn.preprocessing import LabelEncoder
#transform categorical data to numeric
labeler = LabelEncoder()
df['RainToday'] = labeler.fit_transform(df['RainToday'])
df['RainTomorrow'] = labeler.fit_transform(df['RainTomorrow'])
df['WindDir9am'] = labeler.fit_transform(df['WindDir9am'])
df['WindDir3pm'] = labeler.fit_transform(df['WindDir3pm'])
df['WindGustDir'] = labeler.fit_transform(df['WindGustDir'])

print(df.head())

#Detect outliars by using IQR
Q1=np.percentile(df['WindGustSpeed'], 75)
Q3=np.percentile(df['WindGustSpeed'], 25)

#IQR = Q3-Q1
IQR = df.WindGustSpeed.describe()['75%'] - df.WindGustSpeed.describe()['25%']
print ("WindGustSpeed IQR : ",IQR)

# Calculate the minimum value and maximum value
min = Q1-1.5*IQR
max = Q3+1.5*IQR
print ("minimum value: ",min)
print ("maximum value: ",max)

plt.boxplot(df.WindGustSpeed,notch=True,vert=False)
plt.show()


Humidity9am_mean=df.Humidity9am.mean()
Humidity3pm_mean=df.Humidity3pm.mean()
Humidity9am_median=df.Humidity9am.median()
Humidity3pm_median=df.Humidity3pm.median()
Humidity9am_std=df.Humidity9am.std()
Humidity3pm_std=df.Humidity3pm.std()
Humidity9am_var=df.Humidity9am.var()
Humidity3pm_var=df.Humidity3pm.var()

print ("Humidity9am_mean : ", Humidity9am_mean)
print ("Humidity3pm_mean : ", Humidity3pm_mean)
print ("Humidity3pm_median : ", Humidity3pm_median)
print ("Humidity9am_median : ", Humidity9am_median)
print ("Humidity9am std : ", Humidity9am_std)
print ("Humidity3pm_std : ", Humidity3pm_std)
print ("Humidity3pm_var  : ", Humidity3pm_var)
print ("Humidity9am var : ", Humidity9am_var)



from sklearn.preprocessing import LabelEncoder
X = df.drop(['RainTomorrow'], axis = 1)
Y = df['RainTomorrow']
#

#
##########Visualization################




#scatterplot matrix
df.hist(bins = 10 , figsize= (14,14))
plt.show()

#histogram for Maxtemp values
sns.histplot(x=df.MaxTemp)
plt.title("MaxTemp Distribution", color="red", fontsize=18)
plt.show()


# correlation heatmap
##using sns
# plt.figure(figsize=(8,8))
# sns.heatmap(df.corr())
# plt.show()

##using go
df_corr = df.corr() # Generate correlation matrix
fig = go.Figure()
fig.add_trace(
    go.Heatmap(
        x = df_corr.columns,
        y = df_corr.columns,
        z = np.array(df_corr)
    )
)
fig.show()


#scatterPlots
scatterPlot= px.scatter(df.sample(2000),
           title='Min Temp. vs Max Temp.',
           x='MinTemp',
           y='MaxTemp',
           color='RainToday')
scatterPlot.show()
#### It shows a linear positive correlation between minimum temperature and maximum temperature


scatterPlot= px.scatter(df.sample(2000),
           title='Humidity vs Temp.',
           x='Humidity3pm',
           y='Temp3pm',
           color='RainTomorrow')
scatterPlot.show()
#### It shows a linear negative correlation between humidity and  temperature


#barPlot
sns.barplot(data=df, x="RainTomorrow", y="Rainfall")
plt.show()
#### The higher the rate of rain, the greater the probability of rain tomorrow

#pieChart
fig = px.pie(df, names='RainToday', title='RainToday',color_discrete_sequence=px.colors.sequential.RdBu)
fig.show()

