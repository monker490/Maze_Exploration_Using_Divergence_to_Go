import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import gaussian_kde
from numpy import linalg as LA
import operator
import random
import matplotlib.pyplot as plt
import Functions as F
from collections import defaultdict
import json
import csv
                  
maze = np.array([[0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],#0
                 [0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],#1
                 [0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],#2
                 [0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],#3
                 [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0],#4
                 [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0],#5
                 [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0],#6
                 [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0],#7
                 [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0],#8
                 [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0],#9
                 [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0],#10
                 [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0],#11
                 [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0],#12
                 [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0],#13
                 [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0],#14
                 [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0],#15
                 [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0],#16
                 [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0],#17
                 [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0],#18
                 [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0]])#19

maze2 = np.array([[0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0],
                  [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0],
                  [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0],
                  [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0],
                  [0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0],
                  [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0],
                  [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0],
                  [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0],
                  [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0],
                  [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0],
                  [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0],
                  [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0],
                  [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0],
                  [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0],
                  [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0],
                  [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0]])

maze3 = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                  [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
                  [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0],
                  [0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0],
                  [0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0],
                  [0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0],
                  [0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0],
                  [0,0,1,1,0,0,1,1,1,0,0,0,1,1,0,0,1,1,0,0],
                  [0,0,1,1,0,0,1,1,1,0,0,0,1,1,0,0,1,1,0,0],
                  [0,0,1,1,0,0,1,1,1,1,1,1,1,1,0,0,1,1,0,0],
                  [0,0,1,1,0,0,1,1,1,1,1,1,1,1,0,0,1,1,0,0],
                  [0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0],
                  [0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0],
                  [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
                  [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])

maze4 = np.array([[0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0],
                  [0,0,1,1,0,1,1,1,0,0,0,0,1,1,1,0,1,1,0,0],
                  [0,0,1,1,0,1,1,1,0,0,0,0,1,1,1,0,1,1,0,0],
                  [0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0],
                  [0,0,1,1,0,1,0,1,1,1,1,1,1,0,1,0,1,1,0,0],
                  [0,0,0,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0,0,0],
                  [1,1,0,1,0,1,1,1,0,1,1,0,1,1,1,0,1,0,1,1],
                  [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
                  [0,0,1,1,0,1,0,1,1,1,1,1,1,0,1,0,1,1,0,0],
                  [0,0,0,0,0,1,0,1,1,1,1,1,1,0,1,0,0,0,0,0],
                  [0,0,1,1,0,1,0,1,1,1,1,1,1,0,1,0,1,1,0,0],
                  [0,0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,1,0,0,0],
                  [1,1,0,1,0,1,0,1,1,1,1,1,1,0,1,0,1,0,1,1],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                  [0,0,1,1,0,1,1,1,0,1,1,0,1,1,1,0,1,1,0,0],
                  [0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,1,0,0,0],
                  [1,1,0,1,0,1,0,1,1,1,1,1,1,0,1,0,1,0,1,1],
                  [0,0,0,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0,0,0],
                  [0,0,1,1,0,1,1,1,0,1,1,0,1,1,1,0,1,1,0,0],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])


actions = [0,1,2,3] # 0 = UP, 1 = RIGHT, 2 = DOWN, 3 = LEFT

N = 2 # number of steps taken to go from 1 state to the next

dtype = 1 #dtype 1 = CS dtype 0 = Euc

GAMMA = 10**-(1/N)

DMAX = 288

K = 1

# alpha = 0.005 #Learnign rate
alpha = 0.01


D0 = K*DMAX/(1 - GAMMA)

# Td = np.ones((20,20,4)) * D0


Td = np.ones((20,20)) * D0

states = []

numiter = 2000

dtg = [D0]

# # View = np.zeros([3,3])
# # center = np.arra[1,1]
# # prev = np.array([[0,0],[0,1],[0,2],[1,0]])
# next = np.transpose(np.array([[1,2],[2,0],[2,1],[2,2]]))

sigxvals = [0.1,0.25,0.5,0.75,1,1.25,1.5]
sigyvals = [0.1,0.25,0.5,0.75,1,1.25,1.5]
similaritywidth = [0.25,0.5,0.75]

# sigxvals = [0.5,1]
# sigyvals = [0.5,1]

# tablevals = namedtuple("Vals",["simw","sigx","sigy"])
table = defaultdict()

for a in similaritywidth:
    for b in sigxvals:
        for c in sigyvals:
            temp = (a,b,c)
            table[str(temp)] = []

# print(table)
# input("")


current = [0,0]
currentaction = np.reshape(np.array([1]),(1,1))

def searchneighbours(state):
    temp = []
    for i in ([-1,0,1]):
        for j in [-1,0,1]:
            if state[0] + i < 0 or state[1] + j < 0 or state[0] + i>19 or state[1] + j>19 or (i == 0 and j ==0):
                continue
            else:
                temp += [[state[0]+i,state[1]+j]]
    x = int(len(temp)/2)
    if (len(temp)%2):
        right = np.array(temp[x+1:])
    else:
        right = np.array(temp[x:])
    left = np.array(temp[:x])
    # print(right)
    # print(left)
    # print("x") 
    return(left,right)
  
    
    
def finddtg(current,A):
    if A == 0:
        next = (current[0]-1, current[1])
    elif A == 1:
        next = (current[0], current[1]+1)
    elif A == 2:
        next = (current[0]+1, current[1])
    else:
        next = (current[0], current[1]-1)
    # next += (A,)
    if (next[0] >= 0 and next[1] >= 0 and next[0] < 20 and next[1] < 20):
        return (next,Td[next])
    else:
        return(next,-9999)
    

def checkreset(current,state):
    if state[0] < 0 or state[1] < 0 or state[0] > 19 or state[1] > 19 or maze4[tuple(state)] == 1:
        return(current)
    else:
        return(state) 

states = set()
# Td = OrigTd
dtg = [D0]

output = []
for curwidth in similaritywidth:
    for cursigx in sigxvals:
        for cursigy in sigyvals:
            for trial in range(10):
                print (trial) 
                for i in range(5000):
                    if len(states) >= 236:
                        break
                    states.add((19-current[0],current[1]))
                    # states += [tuple(current)]
                    distl,distr = searchneighbours(current)
                    D = F.divergence(distl,distr,cursigx,cursigy,curwidth)
                    # print(np.append(distl,distr,axis = 0))
                    D = D[dtype]
                    statedtg = list(finddtg(current,a) for a in actions)
                    # print(statedtg)
                    maxdiv = max(statedtg,key=operator.itemgetter(1))[1]
                    temp2 = []
                    for s in statedtg:
                        # print(s)
                        if maxdiv == s[1]:
                            temp2+=[s]
                    update = random.choice(temp2)
                    nextaction = statedtg.index(update)
                    nextaction = np.reshape(np.array(nextaction),(1,1))
                    # print(nextaction)
                    # print(update)
                    nextstate = list(update[0][:2])
                    currdtg = D + GAMMA*update[1] - dtg[i]
                    centropy,cmean = F.entropy(np.append(distl,distr,axis = 0),cursigx)
                    nextl,nextr = searchneighbours(nextstate)
                    nentropy,nmean = F.entropy(np.append(nextl,nextr,axis = 0),cursigy)
                    
                    # print(np.append(distl,distr))
                    augmented  = rbf_kernel(np.reshape(current,(1,2)),np.reshape(nextstate,(1,2)))*rbf_kernel(currentaction,nextaction)
                    augmented *= rbf_kernel(centropy,nentropy)*rbf_kernel(cmean,nmean)
                    product = np.squeeze(alpha*currdtg*augmented)
                    # product = np.squeeze(alpha*currdtg*rbf_kernel(np.reshape(current,(1,2)),np.reshape(nextstate,(1,2)))) #this is for traditional dtg
                    
                    dtg += [dtg[i] + product]
                    Td[update[0]] = dtg[-1]
                    current = checkreset(current,nextstate)
                    currentaction = nextaction
                    # print(current)
                    # print("-----")
                tempdata = (curwidth,cursigx,cursigy)
                table[str(tempdata)] += [(i,len(states),dtg[-1])]
                output += [i]    
                # print(dtg[-1])
                # print(len(states))
                # print(output)
                states = set()
                Td = np.ones((20,20)) * D0
                dtg = [D0]
                current = [0,0]


resultFile = open("maze4.csv",'w')
wr = csv.writer(resultFile, dialect='excel')
for keys in table:
    first = []
    second = []
    third = []
    for i in table[keys]:
        first += [i[0]]
        second += [i[1]]
        third += [i[2]]
    tempmean = np.mean(first)
    first += [(tempmean,"mean"),(min(first),"min"),(max(first),"max"),(tempmean/236,"rate exploration")]
    third += [(np.mean(third),'avg_dtg_val')]
    first.insert(0,", ".join((keys,"steps")))
    second.insert(0,", ".join((keys,"states")))
    third.insert(0,", ".join((keys,"dtg_vals")))
    wr.writerow(map(lambda x: [x],first))
    wr.writerow(map(lambda x: [x],second))
    wr.writerow(map(lambda x: [x],third))