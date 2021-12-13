import os

#numtemplate limit
#set1 3
#set2 5
#set3 10

command1 = "python main.py --setnum 1 --numModel 1 --numTemplate 3"
os.system(command1)

command2 = "python scorer/scoring_script.py"
os.system(command2)

