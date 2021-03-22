import matplotlib.pyplot as plt
import torchvision

test= open('accgg-32.txt')
epoch=[]
acc=[]
i=1
for f in test:
    acc.append(float(f[20:25]))
    epoch.append(i)
    i+=1

train=open('loggg-32.txt')
j=1
acc_t=[]
for k in train:
    if j%3125 ==0:
        if k[30].isdigit():
            acc_t.append(float(k[30:35]))
        else:
            acc_t.append(float(k[31:36]))
    j+=1
"""
test1=open('ci-acc3.txt')
acc1=[]
i=1
for f in test1:
    acc1.append(float(f[20:25]))
    i += 1
test2 = open('ci-accml.txt')
i=1
acc2=[]
for f in test2:
    acc2.append(float(f[20:25]))
    i+=1
test3 = open('ci-acc16.txt')
i=1
acc3=[]
for f in test3:
    acc3.append(float(f[20:25]))
    i+=1
"""
plt.figure()
plt.plot(epoch,acc_t,label='training accuracy')
plt.plot(epoch,acc,label='validation accuracy',linewidth=0.5)
"""
plt.plot(epoch,acc3,label='original',linewidth=1)

plt.plot(epoch,acc1,label='3*3conv',linewidth=1)
plt.plot(epoch,acc2,label='3*3conv+multilr',linewidth=1)
plt.plot(epoch,acc,label='3*3conv+multilr+auxloss',linewidth=1)
"""
plt.legend()
plt.grid()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('tiny-imagenet on ggnet with 32batchsize')
#plt.title('results of 4 kinds of training method')
plt.axis([0,100,0,100])
plt.show()