# -*- coding: utf-8 -*-
"""
Created on Sun May  9 11:29:44 2021

@author: xull
"""

from matplotlib import pyplot as pl
import numpy as np

games = [
    'HalfCheetah-v2',
    'Walker2d-v2',
    'Swimmer-v2',
    'Hopper-v2',
    'Reacher-v2',
    'Ant-v2',
    'InvertedPendulum-v2',
    'InvertedDoublePendulum-v2',
]

file = "VARAC-NPG-Ant.txt"

with open(file) as f:
    lines = f.readlines()


count = -1
steps = []
result = []
delta = [1e-1, 1e-2,1e-3,1e-4]
lr =  [1e-2, 1e-3, 1e-4]

ppo_clip_ratio = [0.2]
lrC = [2.5e-2, 2.5e-3, 2.5e-4]
lrA = [2.5e-2, 2.5e-3, 2.5e-4]

# delta = [5e-3, 1e-3, 5e-4]
# lr =  [5e-2, 1e-2, 5e-3]

x = len(ppo_clip_ratio)
y = len(lrC)
z = len(lrA)
m = len(delta)
n = len(lr)
t = 5
for line in lines:
    if "steps 0" in line and "steps/s" in line:

        if count >= 0:
            pl.plot(steps, result)
            pl.xlabel("steps")

            pl.ylabel("episodic return")
            pl.xticks([0, 1e6], ['0', r'$10^6$'])

            # a = count // (y*z)
            # b = count % (y*z) // z
            # c = count % z
            # pl.title(games[t]+", ppo_clip_ratio = " + str(ppo_clip_ratio[a])+", lrC = " + str(lrC[b])+", lrA = " + str(lrA[c]))
            e = count // (m*n)
            a = count % (m*n)// n
            b = count % n
            pl.title(games[t]+", delta = " + str(delta[a])+", lr = " + str(lr[b]))
            pl.show()
        steps = []
        result = []
        count += 1
        
        
    elif "episodic_return_test" in line:
        start = line.find("steps ") + len("steps ")
        end = line.find(", episodic")
        steps.append(int(line[start:end]))
        start = line.find("_test") + len("_test")
        end = line.find("(")
        result.append(float(line[start:end]))

    # if count == 3:
    #     break
pl.plot(steps, result)
pl.xlabel("steps")

pl.ylabel("episodic return")
pl.xticks([0, 1e6], ['0', r'$10^6$'])
# a = count // (y*z)
# b = count % (y*z) // z
# c = count % z
# pl.title(games[t]+", ppo_clip_ratio = " + str(ppo_clip_ratio[a])+", lrC = " + str(lrC[b])+", lrA = " + str(lrA[c]))
e = count // (m*n)
a = count % (m*n)// n
b = count % n
pl.title(games[t]+", delta = " + str(delta[a])+", lr = " + str(lr[b]))
# pl.title(games[count])
pl.show()

