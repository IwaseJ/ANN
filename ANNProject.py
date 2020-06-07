import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data_info = pd.read_csv('D:\\Python\\DeepLearning\\Tenserflow Deep Learning\\DATA\\lending_club_info.csv',index_col='LoanStatNew')
print(data_info.loc['revol_util']['Description'])

def feat_info(col_name):
    print(data_info.loc[col_name]['Description'])

#feat_info('mort_acc')


df = pd.read_csv('D:\\Python\\DeepLearning\\Tenserflow Deep Learning\\DATA\\lending_club_loan_two.csv')
df.info()


sns.countplot(x='loan_status', data=df)

plt.figure(figsize=(12,4))
sns.distplot(df['loan_amnt'],kde=False,bins=40)
plt.xlim(0,45000)

dfcorr= pd.DataFrame(df.corr())

plt.figure(figsize=(12,7))
sns.heatmap(dfcorr, annot=True, cmap='viridis')
plt.ylim(10,0)

#feat_info('installment')
#feat_info('loan_amnt')

plt.figure(figsize=(12,6))
sns.scatterplot(x='installment', y='loan_amnt', data=df)


sns.boxplot(x='loan_status', y='loan_amnt', data=df)

df.groupby('loan_status')['loan_amnt'].describe()

sorted(df['grade'].unique())
sorted(df['sub_grade'].unique())

sns.countplot(x='grade', data=df, hue = 'loan_status')


plt.figure(figsize=(12,4))
subgrade_order = sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade',data=df,order = subgrade_order,palette='coolwarm' )

plt.figure(figsize=(12,4))
subgrade_order = sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade',data=df, order = subgrade_order,palette='coolwarm', hue='loan_status')

f_and_g = df[(df['grade']=='G') | (df['grade']=='F')]
plt.figure(figsize=(12,4))
subgrade_order = sorted(f_and_g['sub_grade'].unique())
sns.countplot(x='sub_grade',data=f_and_g,order = subgrade_order,hue='loan_status')

df['loan_status'].unique()
df['loan_repaid'] = df['loan_status'].map({'Fully Paid':1,'Charged Off':0})
df[['loan_repaid','loan_status']]

df.corr()['loan_repaid'].sort_values().drop('loan_repaid').plot(kind='bar')

head=pd.DataFrame(df.head())
len(df)

missing_data = pd.DataFrame(df.isnull().sum(axis=0))

missing_data_perc = 100* df.isnull().sum()/len(df)

#feat_info('emp_title')
#feat_info('emp_length')

df['emp_title'].nunique()
df['emp_title'].value_counts()

df = df.drop('emp_title',axis=1)

sorted(df['emp_length'].dropna().unique())

emp_length_order = [ '< 1 year',
                     '1 year',
                     '2 years',
                     '3 years',
                     '4 years',
                     '5 years',
                     '6 years',
                     '7 years',
                     '8 years',
                     '9 years',
                     '10+ years']

plt.figure(figsize=(12,4))
sns.countplot(x='emp_length',data=df,order=emp_length_order)

plt.figure(figsize=(12,4))
sns.countplot(x='emp_length',data=df,order=emp_length_order, hue = 'loan_status')

emp_co = df[df['loan_status']=="Charged Off"].groupby("emp_length").count()['loan_status']
emp_fp = df[df['loan_status']=="Fully Paid"].groupby("emp_length").count()['loan_status']
emp_len = emp_co/emp_fp
emp_len
emp_len.plot(kind='bar')

df = df.drop('emp_length',axis=1)
df.isnull().sum()

df['purpose'].head(10)
df['title'].head(10)

df = df.drop('title',axis=1)

#feat_info('mort_acc')


df['mort_acc'].value_counts()

print("Correlation with the mort_acc column")
df.corr()['mort_acc'].sort_values()

#feat_info('total_acc')

df['total_acc'].value_counts()

df.groupby('total_acc').mean()['mort_acc']

total_acc_avg = df.groupby('total_acc').mean()['mort_acc']
total_acc_avg[2.0]

def fill_mort_acc(total_acc, mort_acc):
  
  if np.isnan(mort_acc):
    return total_acc_avg[total_acc]
  else:
    return mort_acc

df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)

df.isnull().sum()

df = df.dropna()

df.isnull().sum()

df.select_dtypes(['object']).columns

#feat_info('term')

df['term'].value_counts()
df['term']=df['term'].apply(lambda term: int(term[:3]))
df['term'].value_counts()

df = df.drop('grade', axis=1)

dummies = pd.get_dummies(['sub_grade'], drop_first=True)
df = pd.concat([df.drop('sub_grade',axis=1),dummies],axis=1)

df.select_dtypes(['object']).columns

dummies = pd.get_dummies(df[['verification_status', 'application_type','initial_list_status','purpose' ]],drop_first=True)
df = df.drop(['verification_status', 'application_type','initial_list_status','purpose'],axis=1)
df = pd.concat([df,dummies],axis=1)

df ['home_ownership'].value_counts()

df['home_ownership'] = df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')

dummies = pd.get_dummies(['home_ownership'], drop_first=True)
df = pd.concat([df.drop('home_ownership',axis=1),dummies],axis=1)
df = pd.concat([df,dummies],axis=1)

df['zip_code'] = df['address'].apply(lambda address:address[-5:])
df['zip_code'].value_counts()

dummies = pd.get_dummies(df['zip_code'],drop_first=True)
df = df.drop(['zip_code','address'],axis=1)
df = pd.concat([df,dummies],axis=1)


df = df.drop('issue_d', axis=1)

df['earliest_cr_line'].apply(lambda date: int(date[-4:]))
df = df.drop('earliest_cr_line', axis =1)
df.select_dtypes(['object']).columns
# Preprocessing

from sklearn.model_selection import train_test_split

df = df.drop('loan_status', axis=1)

X = df.drop('loan_repaid', axis = 1).values
y= df['loan_repaid'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# Model

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()

model.add(Dense(40, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(20, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')

model.fit(x=X_train, y=y_train, epochs=25, batch_size=256,
          validation_data=(X_test, y_test))



