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
    # 'Reacher-v2',
    # 'Ant-v2',
    'InvertedPendulum-v2',
    'InvertedDoublePendulum-v2',
]


methods = ["STOPS-PPO", "MVPI-PPO","VARAC-PPO"]
# methods = ["TOPS-tanh","TOPS-relu", "VARAC-tanh", "VARAC-relu"]
# methods = ["TOPS-tanh","TOPS-relu"]

# methods = ["STOPS-NPG","MVPI-NPG", "VARAC-NPG", "MVP"]

x = len(games)
y = len(methods)
run = 10
# for i in range(x*y):
#     result.append([])
#     steps.append([])
for t in range(x):
    steps = []
    result = []
    count = -1

    file = "PPO-"+games[t]+".txt"
    with open(file) as f:
        lines = f.readlines()
    
    for line in lines:
    
        if "steps 0" in line and "steps/s" in line:
            count += 1
            if count%run == 0:
                steps.append([])
                result.append([])
            line_count = 0
    
    
        elif "episodic_return_test" in line:
            start = line.find("_test") + len("_test")
            end = line.find("(")
            if count%run == 0:
                result[-1].append([])
                result[-1][-1].append(float(line[start:end]))
                start = line.find("steps ") + len("steps ")
                end = line.find(", episodic")
                steps[-1].append(int(line[start:end]))
            else:
                result[-1][line_count].append(float(line[start:end]))
            line_count += 1
    
    for i in range(y):
        j = i
        # j = n*len(methods)+i
        if run == 1:
                pl.plot(steps[j], result[j], label = methods[i])
        else:
            mean = np.mean(result[j],axis=1)
            std = np.std(result[j],axis=1)
            pl.plot(steps[j], mean, label = methods[i])
            pl.fill_between(steps[j], mean-std, mean+std, alpha=0.5)
    
    pl.xlabel("steps", fontsize=15)
    
    pl.ylabel("episodic return", fontsize=20)
    if t == 4:
        pl.legend(loc="lower right")
    else:
        pl.legend(loc="upper left")
    # 
    pl.xticks([0, 1e6], ['0', r'$10^6$'])
    pl.title(games[t])
    
    plfile = "PPO-"+games[t]
    pl.savefig(plfile+".svg")
    pl.savefig(plfile+".png")
    pl.show()




# run = 10
# file="HalfCheetah-v2-EOT_eval_100-lrA_0.0003-lrC_0.001-ppo_ratio_clip_0.2-remark_mvpi_ppo-run-0-211006-235732.txt"
# with open(file) as f:
#     lines = f.readlines()


# n = -y
# for line in lines:

#     if "steps 0" in line and "steps/s" in line:
#         count += 1
#         if count%run == 0:
#             n = n+y
#         line_count = 0


#     elif "episodic_return_test" in line:
#         start = line.find("_test") + len("_test")
#         end = line.find("(")
#         if count%run == 0:
#             result[n].append([])
#             result[n][-1].append(float(line[start:end]))
#             start = line.find("steps ") + len("steps ")
#             end = line.find(", episodic")
#             steps[n].append(int(line[start:end]))
#         else:
#             result[n][line_count].append(float(line[start:end]))
#         line_count += 1

# run = 5
# file = "HalfCheetah-v2-EOT_eval_100-lrA_0.003-lrC_0.001-ppo_ratio_clip_0.2-remark_mops_ppo-run-0-211005-221553.txt"
# with open(file) as f:
#     lines = f.readlines()
# pos = [2,1]
# n = -1
# count = -1
# for line in lines:

#     if "steps 0" in line and "steps/s" in line:
#         count += 1
#         if count%run == 0:
#             n = n + pos[count%2]
#         line_count = 0


#     elif "episodic_return_test" in line:
#         start = line.find("_test") + len("_test")
#         end = line.find("(")
#         if count%run == 0:
#             result[n].append([])
#             result[n][-1].append(float(line[start:end]))
#             start = line.find("steps ") + len("steps ")
#             end = line.find(", episodic")
#             steps[n].append(int(line[start:end]))
#         else:
#             result[n][line_count].append(float(line[start:end]))
#         line_count += 1

# file = "Walker2d-v2-EOT_eval_100-lrA_0.0003-lrC_0.001-ppo_ratio_clip_0.2-remark_mops_ppo-run-0-211007-150902.txt"
# with open(file) as f:
#     lines = f.readlines()
# pos = [4,5,10,11,13,14,16,17]
# count = -1
# for line in lines:

#     if "steps 0" in line and "steps/s" in line:
#         count += 1
#         if count%run == 0:
#             n = pos[count//run]
#         line_count = 0


#     elif "episodic_return_test" in line:
#         start = line.find("_test") + len("_test")
#         end = line.find("(")
#         result[n][line_count].append(float(line[start:end]))
#         line_count += 1


# result[1] = []
# result[7] = []

# file = "HalfCheetah-v2-EOT_eval_100-lrA_0.0003-lrC_0.0001-ppo_ratio_clip_0.2-remark_mops_ppo-run-0-211007-202522.txt"
# with open(file) as f:
#     lines = f.readlines()
# pos = [1,2,7,8]
# count = -1
# for line in lines:

#     if "steps 0" in line and "steps/s" in line:
#         count += 1
#         if count%run == 0:
#             n = pos[count//run]
#         line_count = 0


#     elif "episodic_return_test" in line:
#         start = line.find("_test") + len("_test")
#         end = line.find("(")
#         if (n==1 or n==7) and count%run == 0:
#             result[n].append([])
#             result[n][-1].append(float(line[start:end]))
#             start = line.find("steps ") + len("steps ")
#             end = line.find(", episodic")
#         else:
#             result[n][line_count].append(float(line[start:end]))
#         line_count += 1

# result[7] = []

# file = "Swimmer-v2-EOT_eval_100-lrA_0.0003-lrC_0.0001-ppo_ratio_clip_0.2-remark_mops_ppo-run-0-211008-053519.txt"
# with open(file) as f:
#     lines = f.readlines()

# count = -1
# n = 7
# for line in lines:

#     if "steps 0" in line and "steps/s" in line:
#         count += 1

#         line_count = 0


#     if "episodic_return_test" in line:
#         start = line.find("_test") + len("_test")
#         end = line.find("(")
#         if count == 0:
#             result[n].append([])
#             result[n][-1].append(float(line[start:end]))
#             start = line.find("steps ") + len("steps ")
#             end = line.find(", episodic")
#         else:
#             result[n][line_count].append(float(line[start:end]))
#         line_count += 1

# file = "HalfCheetah-v2-EOT_eval_100-lrA_0.0003-lrC_0.0001-ppo_ratio_clip_0.2-remark_mops_ppo-run-0-211008-091658.txt"
# with open(file) as f:
#     lines = f.readlines()
# pos = [1,7]
# count = -1

# for line in lines:

#     if "steps 0" in line and "steps/s" in line:
#         count += 1
#         if count%run == 0:
#             n = pos[count//run]
#         line_count = 0


#     if "episodic_return_test" in line:
#         start = line.find("_test") + len("_test")
#         end = line.find("(")
#         result[n][line_count].append(float(line[start:end]))
#         line_count += 1


# for n in range(x):
#     for i in range(y):
#         # j = i
#         j = n*len(methods)+i
#         if run == 1:
#                 pl.plot(steps[j], result[j], label = methods[i])
#         else:
#             mean = np.mean(result[j],axis=1)
#             std = np.std(result[j],axis=1)
#             pl.plot(steps[j], mean, label = methods[i])
#             pl.fill_between(steps[j], mean-std, mean+std, alpha=0.5)
    
#     pl.xlabel("steps")
    
#     pl.ylabel("episodic return")
#     pl.legend(loc="upper left")
#     pl.xticks([0, 1e6], ['0', r'$10^6$'])
#     pl.title(file.split('-')[0])
#     pl.title(games[n])
#     pl.show()


    
# file = "Swimmer-v2-EOT_eval_100-lrA_0.0003-lrC_0.0001-ppo_ratio_clip_0.2-remark_mops_ppo-run-0-211008-053519.txt"
# with open(file) as f:
#     lines = f.readlines()

# for line in lines:

#     if "steps 0" in line and "steps/s" in line:
#         count += 1
#         if count%run == 0:
#             steps.append([])
#             result.append([])
#         line_count = 0


#     elif "episodic_return_test" in line:
#         start = line.find("_test") + len("_test")
#         end = line.find("(")
#         if count%run == 0:
#             result[-1].append([])
#             result[-1][-1].append(float(line[start:end]))
#             start = line.find("steps ") + len("steps ")
#             end = line.find(", episodic")
#             steps[-1].append(int(line[start:end]))
#         else:
#             result[-1][line_count].append(float(line[start:end]))
#         line_count += 1
#     methods = ["MVP", "MVPI-NPG", "MOPS-NPG"]
#     n = len(methods)

#     for j in range(n):
#         results[i].append([])
#         for k in range(len(result[i][j])):
#             a = []
#             p = j
#             while p < 30:
#                 a.append(result[i][p][k])
#                 p += 3
#             array = np.array(a)
#             results[i][j].append(a)
#     f.close()



# for i in range(len(games)):
#     file = games[i]+"-VARAC.txt"
    
#     with open(file) as f:
#         lines = f.readlines()

#     # steps[i].append([])
#     results[i].append([])
#     count = 0

#     for line in lines:
    
#         if "steps 0" in line and "steps/s" in line:
               
    
#             count = 0
#             steps[i].append([])

#         elif "episodic_return_test" in line:
            
#             start = line.find("steps ") + len("steps ")
#             end = line.find(", episodic")
#             steps[i][-1].append(int(line[start:end]))
#             start = line.find("_test") + len("_test")
#             end = line.find("(")
#             if len(results[i][-1]) <= count:
#                 results[i][-1].append([])
#             results[i][-1][count].append(float(line[start:end]))
#             count += 1


    # for j in range(n):
    #     results[i].append([])
    #     for k in range(len(result[i][j])):
    #         a = []
    #         p = j
    #         while p < 30:
    #             a.append(result[i][p][k])
    #             p += 3
    #         array = np.array(a)
    #         results[i][j].append(a)
    # f.close()
# new_m="VARAC-NPG"

# methods.append(new_m)
# for j in range(len(games)):
#     steps[j][3]=steps[j][30]
#     for i in range(len(methods)):
#         mean = np.mean(results[j][i],axis=1)
#         std = np.std(results[j][i],axis=1)
#         pl.plot(steps[j][i], mean, label = methods[i])
#         pl.fill_between(steps[j][i], mean-std, mean+std, alpha=0.5)

#     pl.xlabel("steps")
    
#     pl.ylabel("episodic return")
#     pl.legend(loc="upper left")
#     pl.xticks([0, 1e6], ['0', r'$10^6$'])
    
#     pl.title(games[j])
#     pl.show()
