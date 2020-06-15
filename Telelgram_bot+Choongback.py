import requests
from konlpy.tag import Okt
from bs4 import BeautifulSoup

## 채팅로그(text) 받고 명사만 따오기
def Finder(text):
    twitter = Okt()
    target = twitter.nouns(text)
    print('최초의 문장 : ', text)
    print('최초의 대상 list : ', target)
    length = len(target)

    ## 리스트(명사) 순회하면서 욕(검출된) 제거

    textnum = 0
    swear = '*발'
    print(length)
    for (first, last) in target:

        if swear in target[textnum]:
            del target[textnum]
            length = length - 1
            break
        else:
            textnum = textnum + 1

    print('욕 제거 후 남은 대상 list : ', target)

    ### 예 외 처 리 & 국 어 사 전

    jun = '준영'
    target_list = ['너', '걔', '우리', jun, '유환',
                   '충석']  # 준영 유환 충석은 채팅장의 참여자 이름으로 여기다가 word2vec사용해서 이름과 관련된 별명 나오면 전처리 하는거 넣으면 ㄱㅊ할듯?
    target_num = 0
    target_num_count = 0

    searchtext = target_list[target_num]
    search = searchtext

    if searchtext in text:
        while target_num < length:

            if target_num == target_num_count:
                if searchtext in text:
                    print("찾은 지칭명사 : " + searchtext)
                    print("You said " + swear + " to " + searchtext)
            target_num = target_num + 1
            target_num_count = target_num_count + 1
        print('예외처리 후 남은 대상 list : ', target)

    else:
        Word_keyword_list = []
        dicnum = 0
        while dicnum < length:
            dicsearch = target[dicnum]
            url = "https://krdict.korean.go.kr/smallDic/searchResult?nationCode=&mainSearchWord=" + dicsearch
            html = requests.get(url)
            soup = BeautifulSoup(html.text, "html.parser")
            word_list = soup.findAll("strong")
            for line in word_list:
                word = line.get_text().split(",")
                for text in word:
                    Word_keyword_list.append(text.strip())

            if dicsearch in Word_keyword_list:
                del target[dicnum]
            dicnum = dicnum + 1
        print('사전처리 후 남은 대상 list : ', target)

        return target[0]

import telegram

from telegram.ext import Updater
from telegram.ext import MessageHandler, Filters
Telegram_token = '888453032:AAEzGw_nzJXG9p5wNniyvyT0aTvrDOriSpI'

bot = telegram.Bot(token=Telegram_token)

updater = Updater(token=Telegram_token)

dispatcher = updater.dispatcher

def Opitimze_message(bot, update):

    my_text=update.message.text
    target =Finder(my_text)

    bot.send_message(chat_id=update.message.chat_id, text=target)


echo_handler = MessageHandler(Filters.text, Opitimze_message)

dispatcher.add_handler(echo_handler)



updater.start_polling()
