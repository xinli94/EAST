import sys
from operator import itemgetter

correct = 0
wrong = 0
items = []
total = float(sys.argv[2])
with open(sys.argv[1]) as f:
    for line in f:
        #answer,score = line.rstrip('\n').split(',')
        score,answer = line.rstrip('\n').split(',')
        if answer == 'yes':
            correct += 1
        elif answer == 'no':
            wrong += 1
        else:
            continue  #inconclusive
       
        items.append([float(score),correct/total,wrong,float(correct)/float(correct+wrong)])

        #print("%s,%g,%d,%g" % (score,correct,wrong,float(correct)/float(correct+wrong)))


items = sorted(items, key=itemgetter(0))

prev = -1
unique_items = []
for item in items:
    thresh = item[0]
    if thresh != prev:
        unique_items.append(item)
    prev = thresh

unique_items = sorted(unique_items, key=itemgetter(0), reverse=True)

for item in unique_items:
    #print("%.16g,%.16g,%d,%.16g" %(item[0],item[1],item[2],item[3]))
    print("%.16g,%g,%g" %(item[0],item[1],item[3]))
