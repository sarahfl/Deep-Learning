import os


def countMaleFemaleFace(path):
    result = []
    for i in os.listdir(path):
        cut = i.split("_")
        result.append(cut[1])

    male = result.count("0")
    female = result.count("1")
    return (male, female)


dir_list = ["animals10", "landscape", "monkey", "natural-images", "UTKFace"]
dir = "C:/Users/sarah/Deep-Learning/data/"

for folder in dir_list:
    path = os.path.join(dir, folder)
    print(folder, len(os.listdir(path)))

    if folder == "UTKFace":
        male, female = countMaleFemaleFace(path)
        print("#maleFace: ", male)
        print("#femaleFace: ", female)


