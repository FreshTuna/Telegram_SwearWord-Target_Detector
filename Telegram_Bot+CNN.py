
CHOSUNGS = [u'ㄱ',u'ㄲ',u'ㄴ',u'ㄷ',u'ㄸ',u'ㄹ',u'ㅁ',u'ㅂ',u'ㅃ',u'ㅅ',u'ㅆ',u'ㅇ',u'ㅈ',u'ㅉ',u'ㅊ',u'ㅋ',u'ㅌ',u'ㅍ',u'ㅎ']
JOONGSUNGS = [u'ㅏ',u'ㅐ',u'ㅑ',u'ㅒ',u'ㅓ',u'ㅔ',u'ㅕ',u'ㅖ',u'ㅗ',u'ㅘ',u'ㅙ',u'ㅚ',u'ㅛ',u'ㅜ',u'ㅝ',u'ㅞ',u'ㅟ',u'ㅠ',u'ㅡ',u'ㅢ',u'ㅣ']
JONGSUNGS = [u'_',u'ㄱ',u'ㄲ',u'ㄳ',u'ㄴ',u'ㄵ',u'ㄶ',u'ㄷ',u'ㄹ',u'ㄺ',u'ㄻ',u'ㄼ',u'ㄽ',u'ㄾ',u'ㄿ',u'ㅀ',u'ㅁ',u'ㅂ',u'ㅄ',u'ㅅ',u'ㅆ',u'ㅇ',u'ㅈ',u'ㅊ',u'ㅋ',u'ㅌ',u'ㅍ',u'ㅎ']
TOTAL = CHOSUNGS + JOONGSUNGS + JONGSUNGS



import requests
from konlpy.tag import Okt
from bs4 import BeautifulSoup
import re


## 채팅로그(text) 받고 명사만 따오기
def Finder(text):
    twitter = Okt()
    target = twitter.nouns(text)
    #print('최초의 문장 : ', text)
    #print('최초의 대상 list : ', target)
    length = len(target)

    ## 리스트(명사) 순회하면서 욕(검출된) 제거

    textnum = 0
    swear = '*발'
    #print(length)
    for (first, last) in target:

        if swear in target[textnum]:
            del target[textnum]
            length = length - 1
            break
        else:
            textnum = textnum + 1

    #print('욕 제거 후 남은 대상 list : ', target)

    ### 예 외 처 리 & 국 어 사 전

    jun = '준*'
    target_list = ['너', '걔', '우리', jun, '유*',
                   '충*']
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


def jamo_split(word, end_char="_"):
    result = []

    for char in word:

        character_code = ord(char)

        if 0xD7A3 < character_code or character_code < 0xAC00:
            result.append(char)
            continue

        chosung_index = int((((character_code - 0xAC00) / 28) / 21) % 19)
        joongsung_index = int(((character_code - 0xAC00) / 28) % 21)
        jongsung_index = int((character_code - 0xAC00) % 28)

        chosung = CHOSUNGS[chosung_index]
        joongsung = JOONGSUNGS[joongsung_index]
        jongsung = JONGSUNGS[jongsung_index]

        # 종성 범위 밖에 있는 것들은 end_char로 메꿔준다.
        if jongsung_index == 0:
            jongsung = end_char

        result.append(chosung)
        result.append(joongsung)
        result.append(jongsung)

    return "".join(result)

def jamo_combine(word):
    result = ""
    index = 0

    while index < len(word):

        try:
            cho = CHOSUNGS.index(word[index]) * 21 * 28
            joong = JOONGSUNGS.index(word[index + 1]) * 28
            jong = JONGSUNGS.index(word[index + 2])

            result += chr(cho + joong + jong + 0xAC00)
            index += 3

        except:
            result += word[index]
            index += 1

    return result

import os
import pandas as pd
import numpy as np

"""
Crawled_data = pd.read_csv("DATA_10.csv",header=None)

Crawled_data.drop_duplicates(inplace=True)


first_column = Crawled_data.columns[0]

Crawled_data = Crawled_data.drop([first_column], axis=1)

Crawled_data = Crawled_data.drop([0])
Crawled_data = Crawled_data.rename(columns={1:0}, inplace=False)


exceptList = re.compile("[^ㄱ-ㅎㅏ-ㅣ가-힣0-9a-zA-Z ]")

def delete_Character(Sentence):
    Sentence = re.sub(exceptList,"", Sentence)
    return Sentence
    
Crawled_data[0] = Crawled_data[0].apply(lambda x: delet_Character(x))

Crawled_data[0] = Crawled_data[0].apply(lambda x: jamo_split(x))
Crawled_data[0] = Crawled_data[0].apply(lambda x: x.split(" "))

sentence_list = list(Crawled_data[0])
len(sentence_list)

# fasttext 적용
from gensim.models import FastText

model = FastText(sentence_list, size=50, window=2, min_count=3, workers=4, sg=1, min_n=3, max_n=6, iter=10)
###
model.save("C:\\Users\\junyoung\\PycharmProjects\\Telegram_Plugin\\fasttext_model")

len(model.wv.vocab)
"""

data = pd.read_csv("badwww.csv", header=None)
print(data)
blank = " "
for i in range(0,3):
    data[i] = data[i].astype("str")
    data[i] = data[i].apply(lambda x: blank if x == 'nan' else x)


data['n-gram'] = data[0] + "$" + data[1] + "$"+ data[2]
data['n-gram'] = data['n-gram'].apply(lambda x: x.split("$"))

data['n-gram'] = data[0] + "$" + data[1] + "$"+ data[2]
del data[0], data[1], data[2] 

data['n-gram'] = data['n-gram'].apply(lambda x: jamo_split(x))
data['n-gram'] = data['n-gram'].apply(lambda x: x.split("$"))

data.head() # 3column이 label이다.

from gensim.models import FastText

embedding_model = FastText.load("C:\\Users\\junyoung\\PycharmProjects\\Telegram_Plugin\\fasttext_model")

data['n-gram'] = data['n-gram'].apply(lambda x: [embedding_model[_] for _ in x])

data.head()

data.to_json("C:\\Users\\junyoung\\PycharmProjects\\Telegram_Plugin\\labeled_data.json")

import pandas as pd
import numpy as np

data = pd.read_json("C:\\Users\\junyoung\\PycharmProjects\\Telegram_Plugin\\labeled_data.json")
data.columns = ["label", "n-gram"]

data['n-gram'] = data['n-gram'].apply(lambda x: (np.array(x).reshape(-1)))


from sklearn.model_selection import train_test_split

y = data.pop('label')
X = data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


X_train = np.array(X_train['n-gram'].tolist())
X_test = np.array(X_test['n-gram'].tolist())
y_train = y_train.values
y_test = y_test.values

X_train.shape  # row개수 x embeedding차원
y_train.shape  # row개수 x 1



def reshape(df, dim):
    return df.reshape(df.shape[0], dim, 1)


X_train = reshape(X_train, 150)
X_test = reshape(X_test, 150)
X_train.shape


# 출처: https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
from keras import backend as K


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
"""

from keras.models import Sequential
from keras.layers import Conv2D, Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras import backend as K

K.clear_session()  # session 초기화

model = Sequential()
model.add(Dense(150, input_shape=(X_train.shape[1], 1), activation="relu"))
model.add(BatchNormalization())
model.add(Conv1D(filters=100, kernel_size=5, activation='relu'))
model.add(Conv1D(filters=100, kernel_size=3, activation='sigmoid'))
model.add(MaxPooling1D(pool_size=3))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1_m])
model.summary()

history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.2)

print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test, y_test)[1]))

import matplotlib.pyplot as plt

epochs = range(1, len(history.history['acc']) + 1)
plt.plot(epochs, history.history['acc'])
plt.plot(epochs, history.history['val_acc'])
plt.plot(epochs, history.history['val_f1_m'])
plt.title('model accuracy')
plt.ylabel('score')
plt.xlabel('epoch')
plt.legend(['train_acc', 'val_acc', 'f1_score'], loc='upper left')
plt.savefig("1DCNN reuslt.png")
plt.show()


model.save("C:\\Users\\junyoung\\PycharmProjects\\Telegram_Plugin\\cnn_model")
"""
# model load
from gensim.models import FastText
from keras.models import load_model
from sklearn.externals import joblib


embedding_model = FastText.load("fasttext_model")
cnn_model = load_model("C:\\Users\\junyoung\\PycharmProjects\\Telegram_Plugin\\cnn_model", custom_objects={'f1_m': f1_m})



import re

patternList = [
    re.compile('((쌍|썅).{0,2}(놈|년))'),
    re.compile('(씨|시).{0,2}(벌|빨|발|바)'),
    re.compile('(병|븅).{0,2}(신|쉰|싄)'),
    re.compile('(좆|존|좃).{0,2}(같|되|는|나|돼)'),
    re.compile('(개|게).{0,2}(같|갓|새|세|쉐|끼)'),
    re.compile('(걸|느).{0,2}(레|금)'),
    re.compile('(꼬|꽂|고).{0,2}(추|츄)'),
    re.compile('(니|너).{0,2}(엄|엠|애|m|M)'),
    re.compile('(애|에).{0,2}(미)'),
    re.compile('(노).{0,2}(애미|앰|엠)'),
    re.compile('(섹|쎅).{0,2}(스|쓰)'),
    re.compile('(ㅅㅂ|ㅄ|ㄷㅊ)'),
    re.compile('(s|S)(e|E)(x|X)'),
    re.compile('(미|뮈|믜).{0,2}(친|췬|칀)'),
    # 마지막에 추가해서 넣은 키워드들
    re.compile('자지|꼴깝|새끼들|애미|짜식|빠굴|씹년|미친넘|18년|폐녀자|미틴|이놈\
               |조센징|미시촌|주접|붕가|패티쉬|쳐먹|뒤질래|쉐리|호로자식|개좌식|뭥미\
               |별창|망나니|딸딸이|니에미|좃|십새|싸보이다|미췬|씨댕|새꺄|쎅스|10세|\
               상넘|꼰대|개놈|꼴갑|시벌탱|씨방새|발기|새끼|10새끼|꼴리|옘병|아구창|\
               개좆|아갈|창녀|염병|포르노|미친놈|음탕|또라이|좃나|한남충|조지다|호로|\
               후빨|조또|지랄|오지구|세끼|슨상님|병쉰|싸가지|빠큐|엠생|시궁창|꼬라지|\
               우라질|혼음|개빡|뒈진|멍청이|뒤진다|어미|듣보|꼴값|광녀|따먹기|양키|\
               잡종|상놈|넌씨눈|떡치기|개년|꼬추|쎄엑|개지랄|18|시부랄|느개비|오짐|\
               보지|부랄|고인물|찌질|정박아|뒤질|개쓰래기|좇같|후려|시키|육갑|씹새|\
               씝창|미쳤나|호모|조온나|씨파|쉬발|십세|병자|게새끼|개새끼|시부럴|개시키|\
               개민폐|언년|쓰발|sex|눈알|뽄세|씹새기|씨팔|앰창|놈|개수작|아가리|무뇌\
               오진다|창놈|좆같|병맛|로리타|그년|씨부럴|저능아|쌔끈|주뎅이|토끼다|대가리|\
               씹팔|디졌|대갈|엠창|트롤|개씹|썅넘|오졌|갈보|씨발|시발|개자식|극혐|개같은|\
               개짱|미친색|기레기|남혐|야설|이새끼|10창|18놈|섹스|씨불|성인체위|십팔|\
               벌레|빠가|운지|빙신|개돼지|장애인|씹창|썩을|꼬붕|매국노|18새끼|발놈|와꾸|\
               느금|허접|고추|미쳤니|노답|오져|같은년|좆까|돌았나|씨빨|새키|븅|좆만|존싫|\
               사이코|십새끼|섹수|조까|시끼|변태새끼|늬미|열폭|년|쥐랄|잡놈|존버|꼴리다|충|\
               자슥|모가지|씨벌탱|빠구리|니앰|싸가지없|쌍|개간|틀니|냄비|씨발년|시부리|쪽바리|\
               저년|씨부랄|씹탱|즤랄|골빈|샹년|젖탱|메갈|시팔|씨빠|쌍년|싸물|싸대기|스트립|\
               좆|씨볼|이씨|이년|이자식|오바|니미랄|새기|후레자식|호구|패드립|에로물|쌍욕|\
               호로놈|5지구|벼엉|찐따|간나|등신|애자|개같이|쓰레기|5지네|니미|뻑큐|좇|\
               개존|관종|빡촌|뒤져|좃밥|엿|귀두|좆나|개짜증|노무|놈현|개쩔|디질|싸죠|\
               씨부리|돌았네|개새|병신|씨바|양놈|쌍놈년|구라|머갈|불륜|성기|에로|년놈|\
               창년|낯짝|자위|불알|썅년|멍텅|오지네|왜놈|아닥|짱깨|이새키|색끼|주뎅|딜도|\
               대갈빡|정신병자|미친|한남|씨방|뻐큐|니미럴|사까시|존만한|꼴통|씨발놈|존나|\
               |홍어|좆나게|후장|섹|놈들|개새키|븽신|개소리|미치|면상|시댕|갈레|돌아이\
               |닥쳐|개같|쌉|정사|쒸벌|고자|좃또|조빠|씹|썅제기랄|버러지|십창|딴년|꺼져|\
               |좇밥|뽄새|눈깔|쪼개|육봉|수간|틀딱|씹쉐|따까리|음란|씹덕|삥땅')]


def return_bad_words_index(content, mode=0):
    # 정규식을 통하여 욕설있는 위치에 *표시 하여 리턴
    if mode == 0:
        for pattern in patternList:
            content = re.sub(pattern, "**", content)
        return content

    else:
        # 문장의 음절과 어절간의 리스트 생성: 어절의 위치를 뽑기 위함
        token_position = []
        token_index = 0
        # 각 캐릭터 위치마다 어절 인덱스 저장
        for char in content:
            token_position.append(token_index)
            if char == " ":
                token_index += 1

    
        badwords = []
        for pattern in patternList:
            matchObjs = re.finditer(pattern, content)
            badwords += [token_position[obj.span()[0]] for obj in matchObjs] 

        content = [" "] + content.split(" ") + [" "] 

        badwords = list(set(badwords))

    return [(content[index], content[index + 1], content[index + 2], index) for index in
            badwords] 

def DETACH(korean_word):
    LIST = []
    for w in list(korean_word.strip()):
        if '가'<=w<='힣':
            ch1 = (ord(w) - ord('가'))//588
            ch2 = ((ord(w) - ord('가')) - (588*ch1)) // 28
            ch3 = (ord(w) - ord('가')) - (588*ch1) - 28*ch2
            LIST.append([CHOSUNGS[ch1], JOONGSUNGS[ch2], JONGSUNGS[ch3]])
        else:
            LIST.append([w])
    return LIST



def chunks(l, n, trigram_list):
    for i in range(0, len(l), n):
        yield (l[i:i + n], trigram_list[i // n][-1])
        
##path
Path = os.path.dirname(os.path.realpath(__file__))
print(Path)
Path_of_file = Path + "\BadWordList.csv"
print(Path_of_file)
if os.path.isfile(Path_of_file):
    print("Yes")
else :
    print("No")
    BadWordList = pd.DataFrame(columns=["username","Sentence","time"])
    BadWordList.to_csv("BadWordList.csv")

def Save_to_csv(i,n,t):
    BadWordList = pd.DataFrame(columns=["username","Sentence","time"])
    BadWordList = BadWordList.append({'username': i,'Sentence': n ,'time': t},ignore_index=True)
    print(BadWordList.head)
    BadWordList.to_csv('BadWordList.csv',mode='a',header=False)


#여기다 자모스플릿 넣어야됨
text="2018년은 충석이의 해이다. 씨발 충빡이는 너무 멋잇다. 충개쌔끼 "


trigram_list = return_bad_words_index(text, mode=1)

trigram_vector = np.array(
        [np.array(embedding_model[jamo_split(word)]) for trigram in trigram_list for word in trigram[:-1]])
trigram_vector = np.array(
    list(chunks(trigram_vector, 3, trigram_list))) 
trigram_vector = np.array(
    [np.append(_[0].flatten(), _[1]) for _ in trigram_vector]) 


word_index = np.int8(trigram_vector[:, -1])
trigram_vector = np.delete(trigram_vector, -1, axis=1)

trigram_vector = trigram_vector.reshape(trigram_vector.shape[0], trigram_vector.shape[1], 1) 

# cnn
print("단어위치\n", word_index)
print("예측 확률 값\n", cnn_model.predict(trigram_vector))
result = cnn_model.predict(trigram_vector) > 0.65 # 0.65보다 높으면 욕설
result = result.reshape(-1).tolist()
print("Class와 단어 위치\n", list(zip(result, word_index.tolist())))

print("비속어\n", np.array(trigram_list)[np.array(result)])
print("비속어 아닌것들\n", np.array(trigram_list)[np.array(result) == False])

import telegram

from telegram.ext import Updater
from telegram.ext import MessageHandler,Filters
Telegram_token = '1254466273:AAG989dbAxaR6vnluttZboqhZvXSM6E_6Uc'

bot = telegram.Bot(token=Telegram_token)

updater = Updater(token=Telegram_token)

dispatcher = updater.dispatcher

def Opitimze_message(bot, update):
    ##bot.send_message(chat_id=update.message.chat_id, text=update.message.text)
    my_text = update.message.text
    if '^^' in my_text:
        print("shiiit")
        my_text = my_text.replace('^^', 'ㅆ')

    #target = Finder(my_text)
    print(type(my_text))
    #bot.send_message(chat_id=update.message.chat_id, text=target)
    print(update.message.text,"is it ok?")
    test = return_bad_words_index(update.message.text, mode=1)
    print(test,"test%%%%")=
    trigram_vector = np.array(
        [np.array(embedding_model[jamo_split(word)]) for trigram in test for word in trigram[:-1]])
    trigram_vector = np.array(
        list(chunks(trigram_vector, 3, test)))
    trigram_vector = np.array(
        [np.append(_[0].flatten(), _[1]) for _ in trigram_vector])

    print(trigram_vector.shape,"trigram shape1 &&&&")

    word_index = np.int8(trigram_vector[:, -1])  
    trigram_vector = np.delete(trigram_vector, -1, axis=1)  

    trigram_vector = trigram_vector.reshape(trigram_vector.shape[0], trigram_vector.shape[1], 1)  

    print(trigram_vector.shape,"trigram shape2 &&&&")
    # Print
    print("단어위치\n", word_index)
    print("예측 확률 값\n", cnn_model.predict(trigram_vector))
    result = cnn_model.predict(trigram_vector) > 0.65 
    print(result)
    if (True in result):
        send_warning = update.message.from_user.username +"님 , 욕설을 그만둬 주세요. "
        bot.send_message(chat_id=update.message.chat_id, text=send_warning)
        t = str(update.message.date.today())

        Save_to_csv(update.message.from_user.username,update.message.text,t)
        print(update.message.chat_id)
        print(update.message.text)
    result = result.reshape(-1).tolist()
    ##print("reeeeesult : ",list(zip(result)),"%%%%%%%%%%%%%%%%%")

    print("Class와 단어 위치\n", list(zip(result, word_index.tolist())))
    print("비속어\n", np.array(test)[np.array(result)])
    print("비속어가 아닌문장\n", np.array(test)[np.array(result) == False])


echo_handler = MessageHandler(Filters.text, Opitimze_message)

dispatcher.add_handler(echo_handler)



updater.start_polling()
