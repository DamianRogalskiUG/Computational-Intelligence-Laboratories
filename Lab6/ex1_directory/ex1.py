import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Load data
titanic_data = pd.read_csv("ex1_directory/titanic.csv")

# Taking only 4 columns
titanic_data = titanic_data[['Class', 'Sex', 'Age', 'Survived']]

# Preparing data for Apriori
titanic_encoded = pd.get_dummies(titanic_data)

# Apriori algorithm
frequent_itemsets = apriori(titanic_encoded, min_support=0.005, use_colnames=True)

# Mining association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.01)
rules = rules.sort_values(by='confidence', ascending=False)

print("Found rules:")
print(rules)

# Visualizing Age vs Survived
plt.figure(figsize=(10, 6))
plt.hist(titanic_data[titanic_data['Survived'] == 'Yes']['Age'], bins=20, alpha=0.5, label='Survived')
# plt.hist(titanic_data[titanic_data['Survived'] == 'No']['Age'], bins=20, alpha=0.5, label='Not Survived')
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age distribution of Survived vs Not Survived')
plt.legend()
plt.savefig("ex1_directory/ex1_age-vs-survival_plot.jpg")
plt.show()

# Visualizing Sex vs Survived
survived_sex = titanic_data[titanic_data['Survived'] == 'Yes']['Sex'].value_counts()
not_survived_sex = titanic_data[titanic_data['Survived'] == 'No']['Sex'].value_counts()
sex_df = pd.DataFrame([survived_sex, not_survived_sex], index=['Survived', 'Not Survived'])
sex_df.plot(kind='bar', stacked=True, figsize=(8, 6))
plt.title('Survival by Gender')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.savefig('ex1_directory/ex1_gender-vs-survival_plot.jpg')
plt.show()
