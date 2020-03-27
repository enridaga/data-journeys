
import numpy as np 
import pandas as pd 
import math
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("../input/independent-data-1222019/Ulterra 5 Blade 150 RPM (Phase 2).csv")
#data.head(58)

#Overall
#../input/independent-data-1222019/Ulterra 4 Blade 100 RPM.csv
#../input/independent-data-1222019/Ulterra 4 Blade 150 RPM.csv
#../input/independent-data-1222019/Ulterra 5 Blade 100 RPM.csv
#../input/independent-data-1222019/Ulterra 5 Blade 150 RPM.csv
#../input/independent-data-1222019/Teratek 4 Blade 90 RPM.csv
#Phase 1
#/kaggle/input/phase-1-1292019/Phase 1/Ulterra 4 Blade 100 RPM (Phase1).csv
#/kaggle/input/phase-1-1292019/Phase 1/Ulterra 4 Blade 150 RPM (Phase1).csv
#/kaggle/input/phase-1-1292019/Phase 1/Ulterra 5 Blade 100 RPM (Phase1).csv
#/kaggle/input/phase-1-1292019/Phase 1/Ulterra 5 Blade 150 RPM (Phase1).csv
#Phase 2
#../input/independent-data-1222019/Ulterra 4 Blade 100 RPM (Phase 2).csv
#../input/independent-data-1222019/Ulterra 4 Blade 150 RPM (Phase 2).csv
#../input/independent-data-1222019/Ulterra 5 Blade 100 RPM (Phase 2).csv
#../input/independent-data-1222019/Ulterra 5 Blade 150 RPM (Phase 2).csv
#../input/independent-data-1222019/Teratek 4 Blade 90 RPM (Phase 2).csv
def Plot_data_seaborn():
    sns.set()
    columns = ['UCS (psi)','WOB (lbf)','RPM','Db (inch)','BR','SR','Blade Count','ROP (ft/hr)']
    sns.pairplot(data[columns], size = 1.5 , kind ='scatter')
    plt.show()

def Turn_data_to_seperate_lists():
    UCS = data['UCS (psi)']
    WOB = data['WOB (lbf)']
    RPM = data['RPM']
    Db = data['Db (inch)']
    BR = data['BR']
    SR = data['SR']
    Nb = data['Blade Count']
    ROP_Data  = data['ROP (ft/hr)']
    
    return UCS, WOB, RPM, Db, BR, SR, Nb, ROP_Data
    

def Kerkar_Model(UCS, WOB, RPM, Db, BR, SR, Nb, w):
    
    #IFA=(Const)/(1+var1*ROP**var2) just an idea.. need this to match our IFA (y-axis) vs DOC (x-axis) trend
    
    IFA = 55 # DONT FORGET TO CHANGE EVERYTIME!
    
    if UCS is 28000:
        Pe = 0
    else:
        Pe = 11000 #for teratek testing
    
    CCS = UCS*(1+w[4]*Pe**w[5]) #Not important if we do not have confining pressures
    ROP = ((w[0]*WOB**w[1]*RPM**w[2]*math.cos(math.radians(SR))) / (CCS**w[3]*Db*math.tan(math.radians(BR+IFA))))
    
    print(CCS) #Use this to ensure output of CCS matches calculated CCS values
    
    return ROP

def Objective_Function(w):
    
    Error = 0
    ROP_pred_list = []
    ROP_pred_list = [Kerkar_Model(UCS, WOB, RPM, Db, BR, SR, Nb, w) for UCS, WOB, RPM, Db, BR, SR, Nb in zip(UCS, WOB, RPM, Db, BR, SR, Nb)]
    Error = [((abs(ROP_Data - ROP_pred))/ROP_Data) for ROP_Data, ROP_pred in zip(ROP_Data, ROP_pred_list)] 
    Ave_Error = sum(Error) / len(ROP_Data)
    return Ave_Error*100


def De_Algorithm(fobj, bounds, mut=0.8, crossp=0.7, popsize=100, its=1000):
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    fitness = np.asarray([fobj(ind) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]
    for i in range(its):
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace = False)]
            mutant = np.clip(a + mut * (b - c), 0, 1)
            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            f = fobj(trial_denorm)
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
                    
                    
        #print("Iteration number= %s" % (i))
        print("Best Fitness= %s" % (fitness[best_idx]))
        #print("Best values= %s" % (best))
        yield best, fitness[best_idx]

def Run_DEA(ite):
    results = []
    result = list(De_Algorithm(Objective_Function, 
                 [(0, 0.0001),         #k1 
                  (2.56, 2.56),         #a1  
                  (0.71, 0.71),      #b1  
                  (0.23, 0.23),          #c1
                 (0,0),            #as 0,0 | CM 0.00288,0.00288
                 (0,0)],           #bs 0,0 | CM 0.673,0.673
                  mut=0.7, crossp=0.8, popsize=15, its=ite))
    
    df = pd.DataFrame(result)
    return results, df


def Best_coffs(df):
    
    df['w1'], df['w2'], df['w3'], df['w4'], df['w5'], df['w6'] = zip(*df[0]) # Unzip
    cols = [0] # Drop the first column
    df.drop(df.columns[cols],axis = 1,inplace = True) # Drop the first column
    df.columns.values[0] = "Fitness" # name the first column as Fitness
    best_coff = df.iloc[len(df)-1,1:] # insert the best coefficients into the best_coff
    
    return best_coff

def Plot_DEA_Evolution(df):
    
    data_ncol=len(df.columns) # number of paramters 
    fig = plt.figure(figsize=(15,15)) # you may change these to change the distance between plots.

    for i in range(1,(data_ncol+1)):
        if i<data_ncol:
            plt.subplot(3, 3, i)
            plt.plot(df['w{}'.format(i)],'bo', markersize=4)
            plt.xlabel('Iteration')
            plt.ylabel('w{}'.format(i))
            plt.grid(True)
        else:       
            plt.subplot(3, 3, data_ncol)
            plt.plot(df['Fitness'],'red', markersize=4)
            plt.xlabel('Iteration')
            plt.ylabel('Fitness')
            plt.grid(True)
    plt.show()

def Plot_variables(x, y, xlab, ylab, xmin, xmax, ymin, ymax):
    
    fig = plt.figure(figsize=(7,7))
    plt.scatter(x, y)
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    plt.xlabel(xlab)
    plt.ylabel(ylab)

### Data visualization
UCS, WOB, RPM, Db, BR, SR, Nb, ROP_Data = Turn_data_to_seperate_lists()
results, df = Run_DEA(1000)
best_coff = Best_coffs(df)
print(best_coff)
Plot_DEA_Evolution(df)
Est_ROP_Model = [Kerkar_Model(UCS, WOB, RPM, Db, BR, SR, Nb, best_coff) for UCS, WOB, RPM, Db, BR, SR, Nb in zip(UCS, WOB, RPM, Db, BR, SR, Nb)]
Est_ROP_Model = pd.DataFrame(Est_ROP_Model)
ROP_Data = pd.DataFrame(ROP_Data)
#Est_ROP_Model.head(20)
#ROP_Data.head(20)
#Plot_variables(ROP_Data, Est_ROP_Model, 'ROP Data ft/hr', 'ROP Model ft/hr', 0, 60, 0, 60)
#Plot_variables(WOB, ROP_Data, 'WOB Data lbf', 'ROP Data ft/hr', 0, 25000, 0, 60)
#Plot_variables(WOB, Est_ROP_Model, 'Model WOB Data lbf', 'Model ROP Data ft/hr', 0, 25000, 0, 60)


#plt.scatter(WOB, ROP_Data)
#plt.plot(WOB, Est_ROP_Model)


#plt.scatter(ROP_Data,Est_ROP_Model)
#Est_ROP_Model.to_csv('Est_ROP_Model_csv_to_submit.csv', index = False)
#Est_ROP_Model['model rop'] = Est_ROP_Model
#Est_ROP_Model
#ROP_Data.to_csv('Est_ROP_Model_csv_to_submit.csv', index = False)
#ROP_Data['rop data'] = ROP_Data
#ROP_Data
