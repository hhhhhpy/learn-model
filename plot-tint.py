import matplotlib.pyplot as plt

val1 = open('accres-50.txt')

acc=[]
for f in val1:
    acc.append(float(f[20:25]))
"""
val2 = open('accgg-2.txt')
for f in val2:
    acc.append(float(f[20:25]))
"""
epoch1 = [i for i in range(47)]

train1=open('logres-50.txt')
acct=[]
j=1
for f in train1:
    if j % 1563 == 0:
        if f[30].isdigit():
            acct.append(float(f[30:35]))
        else:
            acct.append(float(f[31:36]))
    j += 1
j=1
"""
train2 = open('loggg-2.txt')
for f in train2:
    if j%196 ==0:
        acct.append(float(f[30:36]))
    j+=1
"""

test= open('accgg-b.txt')
epoch=[]
acc_1=[]
i=1
for f in test:
    acc_1.append(float(f[20:25]))
    epoch.append(i)
    i+=1

train=open('loggg-b.txt')
j=1
acc_t=[]
for k in train:
    if j%391 ==0:
       acc_t.append(float(k[30:36]))
    j+=1
plt.figure()

plt.plot(epoch1,acct,label='training accuracy')
plt.plot(epoch1,acc,label='validation accuracy',linewidth=0.5)
#plt.plot(epoch,acc_t,'b',label='training accuracy-256')
#plt.plot(epoch,acc_1,'b',label='validation accuracy-256',linewidth=0.5)
plt.legend()
plt.grid()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('tiny-imagenet on resnet50')

plt.axis([0,50,0,100])
plt.show()