import pandas as pd
import numpy as np
import math
from fuzzywuzzy import process


missing_values = ["NA", "-", "nan"]
columns = ["sepal.length", "sepal.width", "petal.length", "petal.width"]
types = ["Setosa", "Versicolor", "Virginica"]
df = pd.read_csv("iris_with_errors.csv", na_values = missing_values)
# print(df.values)

nullsAmount = df.isnull().sum()
print(f"Nieuzupełnione dane:\n{nullsAmount}")

columnsMedian = {}
for column in columns:
    columnsMedian[column] = df[column].median()

print(columnsMedian)

i = 1
for row in df.values:
    for value in row:
        if not i % 5 == 0:
            if math.isnan(value):
                value = columnsMedian[columns[i % 5 - 1]]
                print(f"Niewypełnione dane. Ustawiono na: {value}")
        else:
            if value not in types:
                print("Błąd w nazwie typu:", value)
                value = process.extractOne(value, types)[0]
                print(f"Poprawione: {value}")
        i += 1

