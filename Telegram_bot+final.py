
CHOSUNGS = [u'ㄱ',u'ㄲ',u'ㄴ',u'ㄷ',u'ㄸ',u'ㄹ',u'ㅁ',u'ㅂ',u'ㅃ',u'ㅅ',u'ㅆ',u'ㅇ',u'ㅈ',u'ㅉ',u'ㅊ',u'ㅋ',u'ㅌ',u'ㅍ',u'ㅎ']
JOONGSUNGS = [u'ㅏ',u'ㅐ',u'ㅑ',u'ㅒ',u'ㅓ',u'ㅔ',u'ㅕ',u'ㅖ',u'ㅗ',u'ㅘ',u'ㅙ',u'ㅚ',u'ㅛ',u'ㅜ',u'ㅝ',u'ㅞ',u'ㅟ',u'ㅠ',u'ㅡ',u'ㅢ',u'ㅣ']
JONGSUNGS = [u'_',u'ㄱ',u'ㄲ',u'ㄳ',u'ㄴ',u'ㄵ',u'ㄶ',u'ㄷ',u'ㄹ',u'ㄺ',u'ㄻ',u'ㄼ',u'ㄽ',u'ㄾ',u'ㄿ',u'ㅀ',u'ㅁ',u'ㅂ',u'ㅄ',u'ㅅ',u'ㅆ',u'ㅇ',u'ㅈ',u'ㅊ',u'ㅋ',u'ㅌ',u'ㅍ',u'ㅎ']
TOTAL = CHOSUNGS + JOONGSUNGS + JONGSUNGS

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

        # 3개의 char를 보아 글자가 만들어지면 만들고 아니면 1개의 char만 추가한다.
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
all_Sentences = pd.read_csv("DATA_10.csv",header=None)

all_Sentences.drop_duplicates(inplace=True)
print(len(all_Sentences))

for col in all_Sentences.columns:
    if col == 0:
        del all_Sentences[col]
        print(all_Sentences)

all_Sentences = all_Sentences.rename(columns={1:0})
all_Sentences = all_Sentences.drop([0])
all_Sentences


# 특수문자나 한자 등이 안날라 갔을 경우를 처리
import re
pattern = re.compile("[^ㄱ-ㅎㅏ-ㅣ가-힣0-9a-zA-Z ]") # 한글 숫자 영어 공백 말고 제거

# 각 단어에 해당 해당 정규표현식 적용
def clear_word(word):
    word = re.sub(pattern, "", word)
    return word

all_Sentences[0] = all_Sentences.astype('str')
all_Sentences[0] = all_Sentences[0].apply(lambda x: clear_word(x))

all_Sentences.reset_index(inplace=True)
all_Sentences.drop('index', axis=1, inplace=True)
all_Sentences = all_Sentences[1:]


all_Sentences[0] = all_Sentences[0].apply(lambda x: jamo_split(x))
all_Sentences[0] = all_Sentences[0].apply(lambda x: x.split(" "))

sentence_list = list(all_Sentences[0])
len(sentence_list)

# fasttext 적용
from gensim.models import FastText
# 임베딩 차원: 50
# window size: 좌우 2단어 비속어는 좌우단어와 별로 연관이 없다고 판단...
# min_count: 최소 3번 등장한 단어들
# workers: -1 전부!!
# sg: skipgram이 더 성능이 좋기 때문
# min_n max_n : n-gram단위인데 한글자가 3글자라 최소 자모3개부터 최대 6개까지 ngram하기로 하였다. 1글자 ~ 2글자
# iter: 반복횟수
model = FastText(sentence_list, size=50, window=2, min_count=3, workers=4, sg=1, min_n=3, max_n=6, iter=10)

model.save("C:\\Users\\junyoung\\PycharmProjects\\Telegram_Plugin\\fasttext_model")

len(model.wv.vocab)

data = pd.read_csv("C:\\Users\\junyoung\\PycharmProjects\\Telegram_Plugin\\badwww.csv", header=None)

# int형으로 문제가 발생하여 전부 str 타입으로 설정
data[0] = data[0].astype("str")
data[1] = data[1].astype("str")
data[2] = data[2].astype("str")

# 빈칸이 nan되는 문제 다시 공백으로 체인지
data[0] = data[0].apply(lambda x: " " if x == 'nan' else x)
data[1] = data[1].apply(lambda x: " " if x == 'nan' else x)
data[2] = data[2].apply(lambda x: " " if x == 'nan' else x)


# 데이터 한 column으로 합치기
data['trigram'] = data[0] + "$" + data[1] + "$"+ data[2]
del data[0], data[1], data[2] # 합친후에 삭제

# 자모분리
data['trigram'] = data['trigram'].apply(lambda x: jamo_split(x))
# ㅂㅏ_ㅂㅗ_ 가 한 word가 될 수 있도록 만들어주는 과정
data['trigram'] = data['trigram'].apply(lambda x: x.split("$"))

data.head() # 3column이 label이다.

# fasttext 모델 불러오기
from gensim.models import FastText

embedding_model = FastText.load("C:\\Users\\junyoung\\PycharmProjects\\Telegram_Plugin\\fasttext_model")

# 각 단어를 벡터화 시켜주는 과정 3 x 50(embedding dimension)
data['trigram'] = data['trigram'].apply(lambda x: [embedding_model[_] for _ in x])

data.head()

data.to_json("C:\\Users\\junyoung\\PycharmProjects\\Telegram_Plugin\\labeled_data.json")















#from JamoSplit import jamo_split

# model load
from gensim.models import FastText
from keras.models import load_model
from sklearn.externals import joblib

import numpy as np

embedding_model = FastText.load("C:\\Users\\junyoung\\PycharmProjects\\Telegram_Plugin\\fasttext_model")
#cnn_model = load_model("C:\\Users\\user\\PycharmProjects\\test\\cnn_model")
rf_model = joblib.load("C:\\Users\\junyoung\\PycharmProjects\\Telegram_Plugin\\rf_model")

text="""

충석이는 요리를 잘한다

"""

import re

# |는 OR 입니다. => (씨 or 시) AND (벌 or 빨 or 발 or 바)
# 예) 쌍1년, 씨12발, 꼬a추
# 중간에 2글자 까지 들어갈 수 있도록 허용하였습니다.

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

    # 비속어 위치와 trigram 리턴
    else:
        # 문장의 음절과 어절간의 리스트 생성: 어절의 위치를 뽑기 위함
        token_position = []
        token_index = 0
        # 각 캐릭터 위치마다 어절 인덱스 저장
        for char in content:
            token_position.append(token_index)
            if char == " ":
                token_index += 1

        # 정규식 표현을 통해 비속어 위치 찾기
        badwords = []
        for pattern in patternList:
            matchObjs = re.finditer(pattern, content)
            badwords += [token_position[obj.span()[0]] for obj in matchObjs]  # 해당 단어가 속한 어절의 위치

        content = [" "] + content.split(" ") + [" "]  # 어절을 반환하기 위한 스플릿 & 맨앞, 맨뒤 padding

        badwords = list(set(badwords))  # 중복제거

    return [(content[index], content[index + 1], content[index + 2], index) for index in
            badwords]  # trigram(3어절 반환) & 단어 위치

#여기다 자모스플릿 넣어야됨



trigram_list = return_bad_words_index(text, mode=1) # 욕설의 형태를 띄는 곳에가서 좌우단어 포함하여 trigram으로 반환


def chunks(l, n, trigram_list):
    '''
    vectroize 할 때 필요한 리스트를 청크별로 나누는 함수
    input : list, n(청크 단위)
    output : (n개씩 묶어진 list, word_index(단어위치))
    '''
    for i in range(0, len(l), n):
        yield (l[i:i + n], trigram_list[i // n][-1])

trigram_list


# vectorize : trigram을 150차원의 벡터 + word index형태의 리스트로 만들어주는 과정
trigram_vector = np.array([np.array(embedding_model[jamo_split(word)]) for trigram in trigram_list for word in trigram[:-1]])
trigram_vector = np.array(list(chunks(trigram_vector, 3, trigram_list))) # 50차원의 3개의 vector가 1개의 trigram에 들어가기위해 나눠주는 과정
trigram_vector = np.array([np.append(_[0].flatten(), _[1]) for _ in trigram_vector]) # 3 x 50 을 150차원으로 flatten + word index = 151dim

trigram_vector.shape

word_index = np.int8(trigram_vector[:, -1]) # word_index 단어위치를 뽑아내기
trigram_vector = np.delete(trigram_vector, -1, axis=1) # word_index 지우기

trigram_vector = trigram_vector.reshape(trigram_vector.shape[0], trigram_vector.shape[1]) # random forest input 맞춰주기

# randomforest
print("단어위치\n", word_index)
print("예측 확률 값\n", rf_model.predict_proba(trigram_vector))
result = rf_model.predict_proba(trigram_vector)[:, 1] > 0.65 # 0.65보다 높으면 욕설
result = result.tolist()
# result = [_==1 for _ in result] # Boolean list로 만들기
print("Class와 단어 위치\n", list(zip(result, word_index.tolist())))


# 결과 확인
print("비속어\n", np.array(trigram_list)[np.array(result)])
print("비속어 아닌것들\n", np.array(trigram_list)[np.array(result) == False])


import telegram

from telegram.ext import Updater
from telegram.ext import MessageHandler,Filters
Telegram_token = '888453032:AAEzGw_nzJXG9p5wNniyvyT0aTvrDOriSpI'

bot = telegram.Bot(token=Telegram_token)

updater = Updater(token=Telegram_token)

dispatcher = updater.dispatcher

def Opitimze_message(bot, update):
    ##bot.send_message(chat_id=update.message.chat_id, text=update.message.text)
    test = return_bad_words_index(update.message.text, mode=1)
    print(test)
    # vectorize : trigram을 150차원의 벡터 + word index형태의 리스트로 만들어주는 과정
    trigram_vector = np.array(
        [np.array(embedding_model[jamo_split(word)]) for trigram in test for word in trigram[:-1]])
    trigram_vector = np.array(
        list(chunks(trigram_vector, 3, test)))  # 50차원의 3개의 vector가 1개의 trigram에 들어가기위해 나눠주는 과정
    trigram_vector = np.array(
        [np.append(_[0].flatten(), _[1]) for _ in trigram_vector])  # 3 x 50 을 150차원으로 flatten + word index = 151dim

    trigram_vector.shape

    word_index = np.int8(trigram_vector[:, -1])  # word_index 단어위치를 뽑아내기
    trigram_vector = np.delete(trigram_vector, -1, axis=1)  # word_index 지우기

    trigram_vector = trigram_vector.reshape(trigram_vector.shape[0],
                                            trigram_vector.shape[1])  # random forest input 맞춰주기

    # randomforest
    print("단어위치\n", word_index)
    print("예측 확률 값\n", rf_model.predict_proba(trigram_vector))
    result = rf_model.predict_proba(trigram_vector)[:, 1] > 0.65  # 0.65보다 높으면 욕설
    if (result[0]):
        bot.send_message(chat_id=update.message.chat_id, text='욕을 그만 두세요 ')
    result = result.tolist()
    ##print("reeeeesult : ",list(zip(result)),"%%%%%%%%%%%%%%%%%")
    # result = [_==1 for _ in result] # Boolean list로 만들기
    print("Class와 단어 위치\n", list(zip(result, word_index.tolist())))

    # 결과 확인
    print("비속어\n", np.array(test)[np.array(result)])
    print("비속어가 아닌문장\n", np.array(test)[np.array(result) == False])


echo_handler = MessageHandler(Filters.text, Opitimze_message)

dispatcher.add_handler(echo_handler)



updater.start_polling()
