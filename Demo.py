import os

#set1 3
#set2 5
#set3 10

command1 = "python main.py --setnum 3 --numModel 1 --numTemplate 1"
os.system(command1)

command2 = "python scorer/scoring_script.py"
os.system(command2)

