import os


path = os.getcwd() + "/train_data"
os.chdir(path)
directory = os.listdir(os.getcwd())
if len(directory) != 2:
    print("Organizing Images")
    os.mkdir("dog")
    os.mkdir("cat")
    for file in directory:
        if file[:3] == "dog":
            os.rename(file, f"dog/{file}")
        elif file[:3] == "cat":
            os.rename(file, f"cat/{file}")
else:
    print("Already Formatted")
