
import numpy as np
import pandas as pd
from numba import njit
import itertools
from multiprocessing import Pool, cpu_count

fam = pd.read_csv('../input/santa-workshop-tour-2019/family_data.csv')
sub = pd.read_csv('../input/santa-ip/submission.csv')
fam = pd.merge(fam,sub, how='left', on='family_id')
choices = fam[['choice_'+str(i) for i in range(10)]].values
fam = fam[['n_people','assigned_day']].values
fam[:1]
fam_costs = np.zeros((5000,101))
for f in range(5000):
    for d in range(1,101):
        l = list(choices[f])
        if d in l:
            if l.index(d) == 0:
                fam_costs[f,d] = 0
            elif l.index(d) == 1:
                fam_costs[f,d] = 50
            elif l.index(d) == 2:
                fam_costs[f,d] = 50 + 9 * fam[f,0]
            elif l.index(d) == 3:
                fam_costs[f,d] = 100 + 9 * fam[f,0]
            elif l.index(d) == 4:
                fam_costs[f,d] = 200 + 9 * fam[f,0]
            elif l.index(d) == 5:
                fam_costs[f,d] = 200 + 18 * fam[f,0]
            elif l.index(d) == 6:
                fam_costs[f,d] = 300 + 18 * fam[f,0]
            elif l.index(d) == 7:
                fam_costs[f,d] = 300 + 36 * fam[f,0]
            elif l.index(d) == 8:
                fam_costs[f,d] = 400 + 36 * fam[f,0]
            elif l.index(d) == 9:
                fam_costs[f,d] = 500 + 235 * fam[f,0]
        else:
            fam_costs[f,d] = 500 + 434 * fam[f,0]
@njit(fastmath=True)
def fclip(p,l=0.):
    for i in range(len(p)):
        if p[i]<l:
            p[i]=l
    return p

@njit(fastmath=True)
def cost_function(pred, p1=1_000_000_000, p2=4000):
    days = np.array(list(range(100,0,-1)))
    daily_occupancy = np.zeros(101)
    penalty = 0
    for i in range(5000):
        penalty += fam_costs[i,pred[i,1]]
        daily_occupancy[pred[i,1]] += pred[i,0]

    for v in daily_occupancy[1:]:
        if (v < 125) or (v >300):
            if v > 300:
                penalty += p1 + abs(v-300)*p2
            else:
                penalty += p1 + abs(v-125)*p2

    penalty += max(0, (daily_occupancy[days[0]]-125.0) / 400.0 * daily_occupancy[days[0]]**(0.5))
    do = daily_occupancy[::-1]
    p = (do[1:] - 125.) / 400. * do[1:] ** (0.5 + ( np.abs(do[1:]-do[:-1]) / 50.0))
    penalty += np.sum(fclip(p))

    return penalty
best = cost_function(fam)
best
@njit(fastmath=True)
def penalty_score_(d, cp, dc):
    penalty = 0
    yc, tc = dc[d + 1], dc[d] + cp #current
    penalty += max(0, (tc-125.0) / 400.0 * tc**(0.5 + abs(tc - yc) / 50.0))
    yc, tc = dc[d] + cp, dc[d -1] #next
    penalty += max(0, (tc-125.0) / 400.0 * tc**(0.5 + abs(tc - yc) / 50.0))
    return penalty

@njit(fastmath=True)
def penalty_score(f,cd,d,cp, dc):
    old = penalty_score_(int(cd), 0, dc) +  penalty_score_(int(d), 0, dc) + fam_costs[f][cd]
    new = penalty_score_(int(cd), -int(cp), dc) +  penalty_score_(int(d), int(cp), dc) + fam_costs[f][d]
    return new - old

@njit(fastmath=True)
def penalty_score2(f1,f2,d1,d2,c1,c2, dc): #single swap - can be improved
    old = penalty_score_(int(d1), 0, dc) +  penalty_score_(int(d2), 0, dc) + fam_costs[f1][d1] + fam_costs[f2][d2]
    new = penalty_score_(int(d1), int(c2-c1), dc) +  penalty_score_(int(d2), int(c1-c2), dc) + fam_costs[f1][d2] + fam_costs[f2][d1]
    return new - old
@njit(fastmath=True)
def optimizer(pred):
    days = np.array(list(range(100,1,-1)))
    days_count = np.zeros(101)
    for i in range(5000):
        days_count[pred[i,1]] += pred[i,0]
    for f in range(5000):
        cd = int(pred[f,1])
        if cd > 1 and cd < 100:
            cp = int(pred[f,0])
            for d in days[1:-1]:
                if d != cd:
                    if days_count[d]+cp>=125 and days_count[d]+cp<=300 and days_count[cd]-cp >= 125 and days_count[cd]-cp<=300:
                        if penalty_score(f, int(cd), int(d), int(cp), days_count)<0:
                            days_count[d] += cp
                            days_count[cd] -= cp
                            pred[f,1] = int(d)
                            cd = int(d)
                        elif fam_costs[f,d] <= fam_costs[f,cd]:
                            dtf = [fx for fx in range(5000) if ((pred[fx,1]==d) and (pred[fx,0]==cp))]
                            for j in dtf: #like for like no move cost
                                if j != f:
                                    if fam_costs[f,d] + fam_costs[j,cd] <= fam_costs[f,cd] + fam_costs[j,d]:
                                        pred[f,1] = int(d)
                                        pred[j,1] = int(cd)
                                        cd = int(d)
                                        #break
    return pred

#https://www.kaggle.com/c/santa-workshop-tour-2019/discussion/119858#latest-687217
@njit(fastmath=True)
def optimizer_a(pred, annealing=5, seed=10):
    np.random.seed(seed)
    days = np.array(list(range(100,1,-1)))
    days_count = np.zeros(101)
    for i in range(5000):
        days_count[pred[i,1]] += pred[i,0]
    for f in range(4999,0,-1):
        cd = int(pred[f,1])
        if cd > 1 and cd < 100:
            cp = int(pred[f,0])
            for d in days[1:-1]:
                if d != cd:
                    if days_count[d]+cp>=125 and days_count[d]+cp<=300 and days_count[cd]-cp >= 125 and days_count[cd]-cp<=300:
                        if penalty_score(f, int(cd), int(d), int(cp), days_count)<  np.random.randint(0, annealing):
                            days_count[d] += cp
                            days_count[cd] -= cp
                            pred[f,1] = int(d)
                            cd = int(d)
                        elif fam_costs[f,d] <= fam_costs[f,cd]:
                            dtf = [fx for fx in range(5000) if ((pred[fx,1]==d) and (pred[fx,0]==cp))]
                            for j in dtf: #like for like no move cost
                                if j != f:
                                    if fam_costs[f,d] + fam_costs[j,cd] <= fam_costs[f,cd] + fam_costs[j,d] + np.random.randint(0, annealing):
                                        pred[f,1] = int(d)
                                        pred[j,1] = int(cd)
                                        cd = int(d)
                                        #break
    return pred

@njit(fastmath=True)
def optimizer_a2(fam, annealing=5., seed=10):
    np.random.seed(seed)
    days_count = np.zeros(101)
    for i in range(5000):
        days_count[fam[i,1]] += fam[i,0]
    for f1 in range(0,5000,1):
        for f2 in range(f1+1,5000,1):
            d1, d2 = int(fam[f1,1]), int(fam[f2,1])
            c1, c2 = int(fam[f1,0]), int(fam[f2,0])
            if f1 != f2 and d1 != d2 and min([d1,d2])>1 and max([d1,d2])<100:
                if days_count[d1]+c2-c1>125 and days_count[d1]+c2-c1<300 and days_count[d2]+c1-c2 > 125 and days_count[d2]+c1-c2<300:
                    if penalty_score2(int(f1), int(f2), int(d1), int(d2), int(c1), int(c2), days_count) <= 0 +  np.random.randint(0, annealing):
                        #print(f1,d1,c1, f2, d2,c2, penalty_score2(int(f1), int(f2), int(d1), int(d2), int(c1), int(c2), days_count))
                        days_count[d2] += c1 - c2
                        days_count[d1] += c2 - c1
                        fam[f1,1] = int(d2)
                        fam[f2,1] = int(d1)
                        d1 = int(d2)
                        #print(cost_function(fam))
    return fam

@njit(fastmath=True)
def optimizer_a3(fam, p1=1_000_000_000, p2=4000):
    for f1 in range(5000):
        for d in range(1,101):
            temp = fam.copy()
            temp[f1,1] = d
            #temp[f1+1,1] = d
            if cost_function(temp,p1,p2) < cost_function(fam,p1,p2):
                #print(f1, d, cost_function(temp)- cost_function(fam))
                fam = temp.copy()
        if f1 % 1000 == 0:
            print('...', f1, cost_function(fam))
    return fam
### %time
best_fam = fam.copy()
for j in range(4):
    fam = optimizer_a3(fam,100,10)
    fam = optimizer_a3(fam,100,100)
    print(j,cost_function(fam))
    for i in range(30,5,-2):
        th = i*10
        df = optimizer_a(fam, i/3, i+1)
        new = cost_function(df)
        #print(i, new, new - best)
        if new <= best + th:
            fam = optimizer_a(df)
            fam = optimizer_a2(fam, 10, i+1)
            fam = optimizer(df) 
            new = cost_function(fam)
            print((j, i), new, new - best)
            if new < best:
                best = new
                best_fam = fam.copy()
fam = optimizer(best_fam)
best = cost_function(fam)
pd.DataFrame({'family_id':list(range(5000)), 'assigned_day':fam[:,1]}).to_csv(f'submission_{best}.csv', index=False)