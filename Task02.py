
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Collection of dataset
titanic = sns.load_dataset('titanic')
print(titanic.head())
print('rows & column:', titanic.shape)

print(titanic.info()) 
print(titanic.isnull().sum())

titanic['age'] = titanic['age'].fillna(titanic['age'].median())


# Numeric column
titanic['age'] = titanic['age'].fillna(titanic['age'].median())

# Categorical columns
titanic['embarked'] = titanic['embarked'].fillna(titanic['embarked'].mode()[0])
titanic['embark_town'] = titanic['embark_town'].fillna(titanic['embark_town'].mode()[0])

# Drop rows with any missing value
titanic.dropna(inplace=True)

# Suppose some 'sex' entries are in capital letters
titanic['sex'] = titanic['sex'].str.lower()

# Convert 'age' to integer
titanic['age'] = titanic['age'].astype(int)

print(titanic.dtypes)

titanic['pclass'] = titanic['pclass'].astype('category')

print(titanic.describe())
print(titanic.describe(include=['O']))
#Grouping
print(titanic.groupby('sex')['survived'].mean())
print(titanic.groupby('pclass')['survived'].mean())

print(pd.crosstab(titanic['sex'], titanic['survived'], normalize='index'))
print(pd.crosstab(titanic['sex'], titanic['survived'], margins='true'))

sns.barplot(x='sex', y='survived', data=titanic)
plt.title('Survival Rate by Gender')
plt.show()

sns.barplot(x='pclass', y='survived', data=titanic)
plt.title('Survival Rate by Passenger Class')
plt.show()

corr = titanic.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()
 
titanic.to_csv('titanic_cleaned.csv', index=False)


