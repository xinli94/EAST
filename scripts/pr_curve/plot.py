import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

colors = ['red', 'green', 'orange', 'blue', 'black']
colors = ['black','blue','green','orange', 'red','brown', 'purple']

for i,d in enumerate(os.listdir(sys.argv[1])):
    if not d.endswith('.roc'):
        continue
    # a = np.loadtxt(os.path.join(sys.argv[1],d),delimiter=',')
    # plt.plot(a[:,1],a[:,2],label=d.split('.')[0], color=colors[i])
    # #plt.plot(a[:,0],a[:,1],label=d.split('.')[0], color=colors[i])

    a = np.loadtxt(os.path.join(sys.argv[1],d),delimiter=',')
    threshold, recall, precision = a[:,0], a[:,1], a[:,2]

    plt.subplot(1,3,1)
    plt.plot(threshold,precision,label=d.split('.')[0], color=colors[i])

    plt.subplot(1,3,2)
    plt.plot(threshold,recall,label=d.split('.')[0], color=colors[i])

    plt.subplot(1,3,3)
    plt.plot(recall,precision,label=d.split('.')[0], color=colors[i])

# plt.legend(loc='lower left')
# plt.title(sys.argv[2])
# plt.xlabel('recall')
# plt.ylabel('precision')
# #plt.xlim((0.0,0.001))
# #plt.xlabel('threshold')
# #plt.ylabel('recall')
# plt.savefig(sys.argv[2] + '.png')
# # plt.show()

plt.subplot(1,3,1)
plt.xlabel('threshold')
plt.ylabel('precision')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), fontsize='xx-small')
plt.axis([-0.05, 1.05, -0.05, 1.05])
plt.gca().set_aspect('equal', adjustable='box')
plt.title('precision/threshold')
# plt.savefig('threshold_precision' + '.png')
# plt.show()

plt.subplot(1,3,2)
plt.xlabel('threshold')
plt.ylabel('recall')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), fontsize='xx-small')
plt.axis([-0.05, 1.05, -0.05, 1.05])
plt.gca().set_aspect('equal', adjustable='box')
plt.title('recall/threshold')
# plt.savefig('threshold_recall' + '.png')
# plt.show()

plt.subplot(1,3,3)
plt.xlabel('recall')
plt.ylabel('precision')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), fontsize='xx-small')
plt.axis([-0.05, 1.05, -0.05, 1.05])
plt.gca().set_aspect('equal', adjustable='box')
plt.title('precision/recall')
# plt.savefig('recall_precision' + '.png')

plt.savefig(sys.argv[2] + '.png')
plt.show()