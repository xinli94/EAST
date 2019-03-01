import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import sys

# colors = ['red', 'green', 'orange', 'blue', 'black']
colors = iter(['black','blue','green','orange', 'red','brown', 'purple'])

files = glob.glob(os.path.join(sys.argv[1], '*.roc'))
# colors = iter(cm.rainbow(np.linspace(0,1,len(files)+1)))

for i, path in enumerate(files):
    d = os.path.basename(path)
    label = os.path.splitext(d)[0]

    a = np.loadtxt(os.path.join(sys.argv[1],d),delimiter=',')
    threshold, recall, precision = a[:,0], a[:,1], a[:,2]

    color = next(colors)

    plt.subplot(1,3,1)
    plt.plot(threshold,precision,label=label, color=color)

    plt.subplot(1,3,2)
    plt.plot(threshold,recall,label=label, color=color)

    plt.subplot(1,3,3)
    plt.plot(recall,precision,label=label, color=color)

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