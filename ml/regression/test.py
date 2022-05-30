from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import math
import numpy as np
from itertools import combinations


#%%
df = pd.read_csv("data/imports-85-clean.csv")
df = df.loc[df["price"] != '?']
df = df.astype({"price": float}, errors='raise')

sample_list = df.drop(columns=["price"]).columns
list_combinations = list()
combos_len = 0
for n in range(len(sample_list) + 1):
    combos = list(combinations(sample_list, n))
    combos_len += len(combos)
    list_combinations += combos

msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]

max_score = 0
best_k = -1
best_c = None
scores = []
p = 1
for c in list_combinations:
    if not c:
        continue
    for i in range(1, 6):
        neigh = KNeighborsRegressor(n_neighbors=i)
        x_train = train[np.asarray(c)]
        y_train = train["price"]
        x_test = test[np.asarray(c)]
        y_test = test["price"]
        neigh.fit(x_train, y_train)

        y_pred = neigh.predict(x_test)
        score = neigh.score(x_test, y_test)
        scores += (c, i, score)
        if score > 0.90:
            print("Tuple: " + str(c) + " | K: " + str(i) + " | Score: " + str(score))
        if score > max_score:
            max_score = score
            best_k = i

    print("Progress: " + str(p) + "/" + str(combos_len) + " - " + ("%.5f%%" % (p/combos_len)))
    p+=1


print(max_score)
print(best_k)


import csv

with open('ur file.csv','wb') as out:
    csv_out=csv.writer(out)
    csv_out.writerow(['attrs','k', 'score'])
    for row in scores:
        if scores[2] > 0.6:
            csv_out.writerow(row)
#%%
