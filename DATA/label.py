import os

import pandas as pd

os.makedirs("NonSeizure", exist_ok=True)
os.makedirs("Seizure", exist_ok=True)

SeizureIndex = 0
NonSeizureIndex = 0

Data = pd.read_csv('EpilepticSeizure.csv', sep=r"\s+", header=None)
for i in range(11500):
    lst = Data.iloc[i].tolist()
    lst = lst[0]
    lst = lst.split(",")
    lst.pop(0)
    a = lst.pop(-1)
    if(str(a) == "1"):
        with open("seizure.txt", "a") as f:
            f.write(",".join(lst))
            f.write("\n")
            f.close()
    else:
        with open("nonseizure.txt", "a") as f:
            f.write(",".join(lst))
            f.write("\n")
            f.close()
