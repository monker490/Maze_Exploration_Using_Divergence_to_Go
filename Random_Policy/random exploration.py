import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import gaussian_kde
from numpy import linalg as LA
import operator
import random
import matplotlib.pyplot as plt
from collections import defaultdict
import csv
import random                  

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


def checkreset(current,state):
    if state[0] < 0 or state[1] < 0 or state[0] > 19 or state[1] > 19 or maze[tuple(state)] == 1:
        return(current)
    else:
        return(state) 


def checkreset2(current,state):
    if state[0] < 0 or state[1] < 0 or state[0] > 19 or state[1] > 19 or maze2[tuple(state)] == 1:
        return(current)
    else:
        return(state) 

def checkreset3(current,state):
    if state[0] < 0 or state[1] < 0 or state[0] > 19 or state[1] > 19 or maze3[tuple(state)] == 1:
        return(current)
    else:
        return(state) 

def checkreset4(current,state):
    if state[0] < 0 or state[1] < 0 or state[0] > 19 or state[1] > 19 or maze4[tuple(state)] == 1:
        return(current)
    else:
        return(state) 

def finder(current,A):
    if A == 0:
        next = (current[0]-1, current[1])
    elif A == 1:
        next = (current[0], current[1]+1)
    elif A == 2:
        next = (current[0]+1, current[1])
    else:
        next = (current[0], current[1]-1)
    return (next)

output = []

for trial in range(100):
    current = [0,0]
    # print(trial)
    states = set()
    for iter in range(100000):
            if len(states) >= 258:
                break
            states.add((19-current[0],current[1]))
            curraction = random.choice(actions)
            nextstate = finder(current,curraction)
            current = checkreset(current,nextstate)
    output += [(iter,len(states))]
    # y,x = zip(*states)
    # print(iter)
    # plt.scatter(x,y)
    # plt.show()
# print (output)

print(2)
output2 = []
for trial in range(100):
    current = [0,0]
    states = set()
    for iter in range(100000):
        if len(states) >= 243:
            break
        states.add((19-current[0],current[1]))
        curraction = random.choice(actions)
        nextstate = finder(current,curraction)
        current = checkreset2(current,nextstate)
    output2 += [(iter,len(states))]
# print (output2)
# y,x = zip(*states)
# print(iter)
# plt.scatter(x,y)
# plt.show()

print(3)
output3 = []
for trial in range(100):
    current = [0,0]
    states = set()
    for iter in range(100000):
        if len(states) >= 226:
            break
        states.add((19-current[0],current[1]))
        curraction = random.choice(actions)
        nextstate = finder(current,curraction)
        current = checkreset3(current,nextstate)
    output3 += [(iter,len(states))]
# print (output3)
# y,x = zip(*states)
# print(iter)
# plt.scatter(x,y)
# plt.show()
print(4)
output4 = []
for trial in range(100):
    current = [0,0]
    states = set()
    for iter in range(100000):
        if len(states) >= 236:
            break
        states.add((19-current[0],current[1]))
        curraction = random.choice(actions)
        nextstate = finder(current,curraction)
        current = checkreset4(current,nextstate)
    output4 += [(iter,len(states))]

# print(output4)

res1 = np.sum(output,axis=0)
res2 = np.sum(output2,axis=0)
res3 = np.sum(output3,axis=0)
res4 = np.sum(output4,axis=0)

avi1 = res1[0]/100
avi2 = res2[0]/100
avi3 = res3[0]/100
avi4 = res4[0]/100

avr1 = res1[0]/res1[1]
avr2 = res2[0]/res2[1]
avr3 = res3[0]/res3[1]
avr4 = res4[0]/res4[1]

min1 = np.amin(output, axis = 0)
min2 = np.amin(output2, axis = 0)
min3 = np.amin(output3, axis = 0)
min4 = np.amin(output4, axis = 0)

max1 = np.amax(output, axis = 0)
max2 = np.amax(output2, axis = 0)
max3 = np.amax(output3, axis = 0)
max4 = np.amax(output4, axis = 0)


print(res1,res2,res3,res4)
print(avi1,avi2,avi3,avi4)
print(avr1,avr2,avr3,avr4)
print(min1,min2,min3,min4)
print(max1,max2,max3,max4)

