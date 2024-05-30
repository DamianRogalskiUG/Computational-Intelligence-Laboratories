import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Load data
titanic_data = pd.read_csv("titanic.csv")

# Taking only 4 columns
titanic_data = titanic_data[['Class', 'Sex', 'Age', 'Survived']]

print(titanic_data)
# Preparing data for Apriori
titanic_encoded = pd.get_dummies(titanic_data)
print(titanic_encoded)

# Apriori algorithm
frequent_itemsets = apriori(titanic_encoded, min_support=0.005, use_colnames=True)
print(frequent_itemsets)

# Mining association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.8)
rules = rules.sort_values(by='confidence', ascending=False)

# Filtering rules
survived_rules = rules[rules['consequents'] == {'Survived_1'}]

# print(survived_rules)

print("Most interesting rules:")
print(survived_rules)


