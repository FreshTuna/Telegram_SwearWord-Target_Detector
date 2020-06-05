import telegram
from telethon import TelegramClient, events
import os
import pandas as pd


cs = list()
cs_date = list()
jy = list()
jy_date = list()
yh = list()
yh_date = list()



Path = os.path.dirname(os.path.realpath(__file__))
print(Path)
Path_of_file = Path + "\BadWordList.csv"

yh_id = 'redstone1618'
jy_id = 'jangjunyoung'
cs_id = 'bluestone1618'
id_list = list()
id_list.append(yh_id)
id_list.append(jy_id)
id_list.append(cs_id)
print(id_list)
BadWordList = pd.read_csv(Path_of_file)
#print(BadWordList.head())
print(len(BadWordList))

for i in range(0,len(BadWordList)):
    id_i=BadWordList['username'][i]
    if str(id_i) in id_list:
        if str(id_i) == yh_id:
            yh.append(BadWordList['Sentence'][i])
            yh_date.append(BadWordList['time'][i])
        elif str(id_i) == jy_id:
            jy.append(BadWordList['Sentence'][i])
            jy_date.append(BadWordList['time'][i])
        elif str(id_i) == cs_id:
            cs.append(BadWordList['Sentence'][i])
            cs_date.append(BadWordList['time'][i])

api_id = 975193
api_hash = '90be99d7814d42c99e5cea19d43701bf'
client = TelegramClient('ad', api_id, api_hash)
client.start()



chii_token = '1254466273:AAG989dbAxaR6vnluttZboqhZvXSM6E_6Uc'
chii = telegram.Bot(token = chii_token)
updates = chii .getUpdates()
for u in updates:
    print(u.message)

import telegram
from telegram.ext import Updater, CommandHandler, MessageHandler,Filters

class TelegramBot:
    def __init__(self, name, token):
        self.core = telegram.Bot(token)
        self.updater = Updater(token)
        self.id = 1012650553
        self.name = name

    def sendMessage(self, text):
        self.core.sendMessage(chat_id = self.id, text=text)

    def stop(self):
        self.updater.start_polling()
        self.updater.dispatcher.stop()
        self.updater.job_queue.stop()
        self.updater.stop()

def optimize_message(bot, update):
    mytext = update.message
    #bot.send_message(chat_id=update.message.chat_id, text=update.message.text)
    print('mytext: ',mytext)

class BotChii(TelegramBot):
    def __init__(self):
        self.token = '1254466273:AAG989dbAxaR6vnluttZboqhZvXSM6E_6Uc'
        TelegramBot.__init__(self, 'bad_bot', self.token)
        self.updater.stop()

    def add_handler(self, cmd, func):
        self.updater.dispatcher.add_handler(CommandHandler(cmd, func))

    def every_message(self):
        echohandler = MessageHandler(Filters.text, optimize_message)
        self.updater.dispatcher.add_handler(echohandler)

    def start(self):
        self.sendMessage('안녕하세요.\n 비속어 탐자기입니다.')
        self.sendMessage('할수있는 명령어')
        self.sendMessage('/css : 충석이가 받은 문장리스트 ')
        self.sendMessage('/jyy : 준영이가 받은 문장리스트 ')
        self.sendMessage('/yhh : 유환이가 받은 문장리스트 ')
        self.updater.start_polling()
        self.updater.idle()

import sys
text = []

current_sentence = ""
@client.on(events.NewMessage)
async def my_event_handler(event):
    await client.get_entity('bottest')
    current_sentence = event.raw_text
    print(current_sentence)
    text.append(current_sentence)

for i in range(len(text)):
    if '충석' in text[i]:
        cs.append(text[i])
    elif '준영' in text[i]:
        jy.append(text[i])
        jy_date.append(text[i])
    elif '유환' in text[i]:
        yh.append(text[i])

def ignite(bot, update):
    bot.send_message(chat_id=update.message.chat_id, text="안녕하세요 \n비속어 탐지기입니다." )
    bot.send_message(chat_id=update.message.chat_id, text="/css : 충석이가 한 문장리스트")
    bot.send_message(chat_id=update.message.chat_id, text="/jyy : 준영이가 한 문장리스트")
    bot.send_message(chat_id=update.message.chat_id, text="/yhh : 유환이가 한 문장리스트")


def css(bot, update):
    bot.send_message(chat_id=update.message.chat_id, text="충석이가 받은 문장리스트")
    # print(str(update.message.date.today()))
    # print(update.message.from_user.username)
    send_text = ""

    for i in range(len(cs)):
        send_text = send_text + str(i + 1) + ". " + cs[i] + " \t " + cs_date[i] + "\n"
    bot.send_message(chat_id=update.message.chat_id, text=send_text)

def jyy(bot, update):
    bot.send_message(chat_id=update.message.chat_id, text="준영이가 한 문장리스트")
    #print(str(update.message.date.today()))
    #print(update.message.from_user.username)
    print(jy)
    send_text=""

    for i in range(len(jy)):
        send_text = send_text + str(i+1) +". " + jy[i]+" \t "+jy_date[i]+"\n"
    bot.send_message(chat_id=update.message.chat_id, text=send_text)

def yhh(bot, update):
    bot.send_message(chat_id=update.message.chat_id, text="유환이가 받은 문장리스트")
    # print(str(update.message.date.today()))
    # print(update.message.from_user.username)
    send_text = ""

    for i in range(len(yh)):
        send_text = send_text + str(i + 1) + ". " + yh[i] + " \t " + yh_date[i] + "\n"
    bot.send_message(chat_id=update.message.chat_id, text=send_text)

def proc_rolling(bot, update):
    chii.sendMessage('할수있는 명령어')
    chii.sendMessage('/css : 충석이가 한 문장리스트 ')
    chii.sendMessage('/jyy : 준영이가 한 문장리스트 ')
    chii.sendMessage('/yhh : 유환이가 한 문장리스트 ')

def proc_stop(bot, update):
    chii.sendMessage('나 요리하러갈게.')
    chii.stop()


current_sentence = list()

chii = BotChii()
chii.add_handler('cbb', proc_rolling)
chii.add_handler('stop', proc_stop)
chii.add_handler('css', css)
chii.add_handler('jyy', jyy)
chii.add_handler('yhh', yhh)
chii.add_handler('command',ignite)
chii.every_message()


chii.start()

client.run_until_disconnected()