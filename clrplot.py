import numpy as np
import matplotlib.pyplot as plt

##定义cyclical_learning_rate，这里为了方便将triangular2和exp_range放在一起来定义
def cyclical_learning_rate(batch_step,
                           step_size,
                           base_lr=0.001,
                           max_lr=0.006,
                           mode='triangular',
                           gamma=0.999995):

    cycle = np.floor(1 + batch_step / (2. * step_size))
    x = np.abs(batch_step / float(step_size) - 2 * cycle + 1)

    lr_delta = (max_lr - base_lr) * np.maximum(0, (1 - x))  #triangular LR

    if mode == 'triangular':
        pass
    elif mode == 'triangular2':
        lr_delta = lr_delta * 1 / (2. ** (cycle - 1))    #triangular2 LR
    elif mode == 'exp_range':
        lr_delta = lr_delta * (gamma**(batch_step))      #exp_range LR
    else:
        raise ValueError('mode must be "triangular", "triangular2", or "exp_range"')

    lr = base_lr + lr_delta

    return lr


##定义超参数
num_epochs = 50     #定义epochs数
num_train = 50000    #定义训练样本
batch_size = 100     #定义batch_size
iter_per_ep = num_train // batch_size   #计算iteration

##triangular可视化
batch_step = -1
collect_lr = []
for e in range(num_epochs):
    for i in range(iter_per_ep):
        batch_step += 1
        cur_lr = cyclical_learning_rate(batch_step=batch_step,
                                        step_size=iter_per_ep*5)

        collect_lr.append(cur_lr)

#plt.scatter(range(len(collect_lr)), collect_lr,linewidth=0.001)
plt.plot(range(len(collect_lr)), collect_lr)
plt.ylim([0.0, 0.01])
plt.xlim([0, num_epochs*iter_per_ep + 5000])
plt.show()