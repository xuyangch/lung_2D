import matplotlib.pyplot as plt

f_list = []
f_list.append(open('/Users/hyacinth/Downloads/肺结节识别/lung_2d_result/auc/lung_2D_aift.out','r'))
f_list.append(open('/Users/hyacinth/Downloads/肺结节识别/lung_2d_result/auc/lung_2D_random.out','r'))
plot_name = ['aift','random']
color = ['r','g']

plt.figure('final')

for idx in range(2):
    label_num = 0
    max_acc = 0.5
    label_list = []
    acc_list = []

    f = f_list[idx]
    name = plot_name[idx]
    for line in f:
        if line.startswith('%d labels'):
            num = int(line.split(' ')[-1])
            if num > label_num:
                label_list.append(label_num)
                acc_list.append(max_acc)
                label_num = num
                max_acc = 0

        if line.startswith('val Loss'):
            acc = float(line.split(' ')[-1])
            if acc > max_acc:
                max_acc = acc

    # plt.figure(name)
    if (idx == 0):
        plt.plot(label_list, acc_list, '-', lw = 1)
    else:
        plt.plot(label_list, acc_list, '-.', lw= 1)
# plt.figure("&".join(plot_name))
plt.xlabel('Labels used')
plt.ylabel('Accuracy')
plt.show()