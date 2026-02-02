import os

path = path = path = "data/train"



for folder in os.listdir(path):
    count = len(os.listdir(os.path.join(path, folder)))
    print(folder, ":", count)
