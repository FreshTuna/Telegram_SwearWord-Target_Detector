#This code is for checking Performance. It is only made for algorithm. It won't work.

count=0
FP=0
TN=0
for i in range(0, len(Choongsuck)):
    print(type(Choongsuck['sentence'][i]))
    try :
        L = Opitimze_message(str(Choongsuck['sentence'][i]))
        if L==Choongsuck['label'][i]:
            count= count+1
        else:
            print("Wroonggggg~!~@~@~@!~@~@!~~@!@")
            if L == 0:
                TN= TN + 1
            else:
                FP = FP + 1

    except:
        print("next")
        if Choongsuck['label'][i]==0:
            count= count+1
        else:
            TN = TN + 1
            print("Wroonggggg~!~@~@~@!~@~@!~~@!@")


print("예측도 : ", count/300,"%")

print("TN : ", TN)

print("FP : ", FP)
