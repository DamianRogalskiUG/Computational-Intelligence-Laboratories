import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("iris.csv")

(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=288493)

def classify_iris(sl, sw, pl, pw):
    if sl > 4:
        return("Setosa")
    elif pl <= 5:
        return("Virginica")
    else:
        return("Versicolor")

good_predictions = 0
len = test_set.shape[0]
for i in range(len):
    sl, sw, pl, pw = train_set[:, 0:4][i][0], train_set[:, 0:4][i][1], train_set[:, 0:4][i][2], train_set[:, 0:4][i][3]
    if classify_iris(sl, sw, pl, pw) == test_set[:, 4][i]:
        good_predictions = good_predictions + 1
        print(good_predictions)
        print(good_predictions/len*100, "%")