import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("iris.csv")
# print(df)

# podzial na zbior testowy (30%) i treningowy (70%), ziarno losowosci = 13
(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=13)

# print(test_set)
# print(test_set.shape[0])
#
train_inputs = train_set[:, 0:4]
# print(train_inputs)
train_classes = train_set[:, 4]
print(train_classes)
# test_inputs = test_set[:, 0:4]
# print(test_inputs)
# test_classes = test_set[:, 4]
# print(test_classes)