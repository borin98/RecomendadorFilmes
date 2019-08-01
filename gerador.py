import matplotlib.pyplot as plt
import pandas as pd
import random as r

"""valid = 1

if valid :

    print(valid)

#fig, ax = plt.subplot()
plt.plot([1,0], label="train")
plt.plot([0,1], label="valid")
leg = plt.legend()

plt.xlabel("Època")
plt.ylabel("Energia Livre")
plt.savefig("FreeEnergy.pdf")"""

df = pd.DataFrame(columns = ["UserId", "MovieId"])

listaUser = []
listaMovie = []

for i in range(1000) :

    listaUser.append(r.randint(1, 5043))
    listaMovie.append(r.randint(1, 5043))

df["UserId"] = listaUser
df["MovieId"] = listaMovie
df.to_csv("testeToWatch.csv")