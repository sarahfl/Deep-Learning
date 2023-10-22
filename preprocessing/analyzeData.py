import os

path = "C:/Users/sarah/Deep-Learning/data/UTKFace"
files = os.listdir(path)

result = []
for i in files:
    cut = i.split("_")
    result.append(cut[1])

male = result.count("0")
female = result.count("1")

# Anteil männlicher und weiblicher Gesichter im UTKFace Datenset
print("male faces: ", male, "female faces: ", female, "total faces: ", len(result))
