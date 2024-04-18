import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("iris.csv")

(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=288493)

def classify_iris(sl, sw, pl, pw):
    if pw < 1 and pl < 2:
        return("Setosa")
    elif sl >= 6 and pl >= 5:
        return("Virginica")
    else:
        return("Versicolor")

good_predictions = 0
len = test_set.shape[0]
print(train_set)
# print(test_set)
for i in range(len):
    print('kolejny')
    sl, sw, pl, pw = test_set[i][:4]
    true_class = test_set[i][4]
    print(classify_iris(sl, sw, pl, pw), sl, sw, pl, pw, true_class)
    if classify_iris(sl, sw, pl, pw) == true_class:
        good_predictions = good_predictions + 1
        print(good_predictions)
        print(good_predictions/len*100, "%")