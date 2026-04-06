import pandas as pd

df = pd.read_csv("../data/patients_dakar.csv")

print("Nombre de patients :", len(df))
print("Colonnes :", df.columns)

print("\nDiagnostics :")
print(df["diagnostic"].value_counts())

print("\nTemp moyenne par diagnostic :")
print(df.groupby("diagnostic")["temperature"].mean())

# Exercice 1
print("\nSexe et diagnostic :")
print(df.groupby(["sexe", "diagnostic"]).size())