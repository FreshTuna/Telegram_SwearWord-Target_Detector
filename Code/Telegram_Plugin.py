#This code is used for Testing Telegram.


import os
import sys

from time import sleep
from telethon.tl.custom.sendergetter import SenderGetter
from telethon.tl.functions.channels import GetFullChannelRequest
from telethon.tl.functions.channels import GetParticipantsRequest
from telethon.tl.types import ChannelParticipantsSearch
from telethon.tl.types import (
PeerChannel
)
from telethon.tl.functions.messages import GetHistoryRequest
from telethon.tl.functions.channels import JoinChannelRequest
from telethon import TelegramClient, events, sync
from telethon.tl.types.contacts import Contacts
from telethon.tl.functions.contacts import GetContactsRequest
from telethon.tl.types import InputPeerUser
from konlpy.tag import Okt

##from Jamosplit  import jamo_split, jamo_combine
CHOSUNGS = [u'ㄱ', u'ㄲ', u'ㄴ', u'ㄷ', u'ㄸ', u'ㄹ', u'ㅁ', u'ㅂ', u'ㅃ', u'ㅅ', u'ㅆ', u'ㅇ', u'ㅈ', u'ㅉ', u'ㅊ', u'ㅋ', u'ㅌ', u'ㅍ',
            u'ㅎ']
JOONGSUNGS = [u'ㅏ', u'ㅐ', u'ㅑ', u'ㅒ', u'ㅓ', u'ㅔ', u'ㅕ', u'ㅖ', u'ㅗ', u'ㅘ', u'ㅙ', u'ㅚ', u'ㅛ', u'ㅜ', u'ㅝ', u'ㅞ', u'ㅟ',
              u'ㅠ', u'ㅡ', u'ㅢ', u'ㅣ']
JONGSUNGS = [u'_', u'ㄱ', u'ㄲ', u'ㄳ', u'ㄴ', u'ㄵ', u'ㄶ', u'ㄷ', u'ㄹ', u'ㄺ', u'ㄻ', u'ㄼ', u'ㄽ', u'ㄾ', u'ㄿ', u'ㅀ', u'ㅁ', u'ㅂ',
             u'ㅄ', u'ㅅ', u'ㅆ', u'ㅇ', u'ㅈ', u'ㅊ', u'ㅋ', u'ㅌ', u'ㅍ', u'ㅎ']
TOTAL = CHOSUNGS + JOONGSUNGS + JONGSUNGS


# 자모분리
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


# 자모결합
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
#okt = Okt()

api_id = 975193
api_hash = '90be99d7814d42c99e5cea19d43701bf'
client = TelegramClient('an', api_id, api_hash)
client.start()

contact_list = {
}

contacts = client(GetContactsRequest(0))
print(contacts)

for u in contacts.users:
     contact_list[u.id] = u.access_hash
     print(contact_list[u.id] ,u.first_name, u.last_name, u.username)

""""
channel ='ASDFZCXV'
channels = {d.entity.username: d.entity
            for d in client.get_dialogs()
            if d.is_channel}

channel = channels[channel]
print(channel)
for u in client.iter_participants(channel, aggressive=True):
  print(u.id, u.first_name, u.last_name, u.username)
"""

current_sentence = ""
banned_word=["참외","포도","사과"]
error_sentence = ""

@client.on(events.NewMessage)
async def my_event_handler(event):
    await client.get_entity('Shit')
    current_sentence = event.raw_text
    #print(okt.pos(current_sentence))
    print(event.message.to_id.chat_id)
    print(event)
    if event.message.to_id.chat_id == 340126983:
        for n in banned_word:
            if n in current_sentence:
                splitted = jamo_split(current_sentence)
                ##error_sentence = "dont say"+ splitted +" please"
                error_sentence = splitted
                merged_sentence = jamo_combine(error_sentence)
                await client.send_message(InputPeerUser(event.message.from_id, contact_list[event.message.from_id]),
                                          error_sentence)
                await client.send_message(InputPeerUser(event.message.from_id, contact_list[event.message.from_id]),
                                          merged_sentence)



client.run_until_disconnected()