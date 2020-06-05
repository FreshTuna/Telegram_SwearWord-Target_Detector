import pandas as pd
import os

Path = os.path.dirname(os.path.realpath(__file__))
print(Path)
Path_of_file = Path + "\BadWordList.csv"
print(Path_of_file)
if os.path.isfile(Path_of_file):
    print("Yes")
else :
    print("No")
    BadWordList = pd.DataFrame(columns=["chatID","Sentence"])
    BadWordList.to_csv("BadWordList.csv")


def Save_to_csv(i,n):
    BadWordList=pd.read_csv('BadWordList.csv')
    BadWordList = BadWordList.append({'chatID': i,'Sentence': n },ignore_index=True)
    print(BadWordList.head)

Save_to_csv(10,"Fuckme")
import telegram

Telegram_token = '888453032:AAEzGw_nzJXG9p5wNniyvyT0aTvrDOriSpI'

bot = telegram.Bot(token=Telegram_token)
updates= bot.getUpdates()

for update in updates:
    print(update)
    lastUpdate_id = update.update_id

