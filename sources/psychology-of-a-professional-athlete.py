
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib as mpl

from matplotlib.patches import Circle, Rectangle, Arc

from sklearn import mixture

from sklearn import ensemble

from sklearn import model_selection

from sklearn.metrics import accuracy_score as accuracy

from sklearn.metrics import log_loss

import time

import itertools

import operator
#%% load training data

allData = pd.read_csv('../input/data.csv')

data = allData[allData['shot_made_flag'].notnull()].reset_index()



#%% add some temporal columns to the data

data['game_date_DT'] = pd.to_datetime(data['game_date'])

data['dayOfWeek']    = data['game_date_DT'].dt.dayofweek

data['dayOfYear']    = data['game_date_DT'].dt.dayofyear



data['secondsFromPeriodEnd']   = 60*data['minutes_remaining']+data['seconds_remaining']

data['secondsFromPeriodStart'] = 60*(11-data['minutes_remaining'])+(60-data['seconds_remaining'])

data['secondsFromGameStart']   = (data['period'] <= 4).astype(int)*(data['period']-1)*12*60 + (data['period'] > 4).astype(int)*((data['period']-4)*5*60 + 3*12*60) + data['secondsFromPeriodStart']



# look at first couple of rows and verify that everything is good

data.loc[:10,['period','minutes_remaining','seconds_remaining','secondsFromGameStart']]
# plot the shot attempts as a function of time (from start of game) with several different binnings

plt.rcParams['figure.figsize'] = (16, 16)

plt.rcParams['font.size'] = 16



binsSizes = [24,12,6]



plt.figure();

for k, binSizeInSeconds in enumerate(binsSizes):

    timeBins = np.arange(0,60*(4*12+3*5),binSizeInSeconds)+0.01

    attemptsAsFunctionOfTime, b = np.histogram(data['secondsFromGameStart'], bins=timeBins)     

    

    maxHeight = max(attemptsAsFunctionOfTime) + 30

    barWidth = 0.999*(timeBins[1]-timeBins[0])

    plt.subplot(len(binsSizes),1,k+1); 

    plt.bar(timeBins[:-1],attemptsAsFunctionOfTime, align='edge', width=barWidth); plt.title(str(binSizeInSeconds) + ' second time bins')

    plt.vlines(x=[0,12*60,2*12*60,3*12*60,4*12*60,4*12*60+5*60,4*12*60+2*5*60,4*12*60+3*5*60], ymin=0,ymax=maxHeight, colors='r')

    plt.xlim((-20,3200)); plt.ylim((0,maxHeight)); plt.ylabel('attempts')

plt.xlabel('time [seconds from start of game]')
#%% plot the accuracy as a function of time

plt.rcParams['figure.figsize'] = (15, 10)

plt.rcParams['font.size'] = 16



binSizeInSeconds = 20

timeBins = np.arange(0,60*(4*12+3*5),binSizeInSeconds)+0.01

attemptsAsFunctionOfTime,     b = np.histogram(data['secondsFromGameStart'], bins=timeBins)     

madeAttemptsAsFunctionOfTime, b = np.histogram(data.loc[data['shot_made_flag']==1,'secondsFromGameStart'], bins=timeBins)     

attemptsAsFunctionOfTime[attemptsAsFunctionOfTime < 1] = 1

accuracyAsFunctionOfTime = madeAttemptsAsFunctionOfTime.astype(float)/attemptsAsFunctionOfTime

accuracyAsFunctionOfTime[attemptsAsFunctionOfTime <= 50] = 0 # zero accuracy in bins that don't have enough samples



maxHeight = max(attemptsAsFunctionOfTime) + 30

barWidth = 0.999*(timeBins[1]-timeBins[0])



plt.figure();

plt.subplot(2,1,1); plt.bar(timeBins[:-1],attemptsAsFunctionOfTime, align='edge', width=barWidth); 

plt.xlim((-20,3200)); plt.ylim((0,maxHeight)); plt.ylabel('attempts'); plt.title(str(binSizeInSeconds) + ' second time bins')

plt.vlines(x=[0,12*60,2*12*60,3*12*60,4*12*60,4*12*60+5*60,4*12*60+2*5*60,4*12*60+3*5*60], ymin=0,ymax=maxHeight, colors='r')

plt.subplot(2,1,2); plt.bar(timeBins[:-1],accuracyAsFunctionOfTime, align='edge', width=barWidth); 

plt.xlim((-20,3200)); plt.ylabel('accuracy'); plt.xlabel('time [seconds from start of game]')

plt.vlines(x=[0,12*60,2*12*60,3*12*60,4*12*60,4*12*60+5*60,4*12*60+2*5*60,4*12*60+3*5*60], ymin=0.0,ymax=0.7, colors='r')
#%% cluster the shot attempts of kobe using GMM on their location

numGaussians = 13

gaussianMixtureModel = mixture.GaussianMixture(n_components=numGaussians, covariance_type='full', 

                                               init_params='kmeans', n_init=50, 

                                               verbose=0, random_state=5)

gaussianMixtureModel.fit(data.loc[:,['loc_x','loc_y']])



# add the GMM cluster as a field in the dataset

data['shotLocationCluster'] = gaussianMixtureModel.predict(data.loc[:,['loc_x','loc_y']])
#%% define draw functions (stealing shamelessly the draw_court() function from MichaelKrueger's excelent script)



def draw_court(ax=None, color='black', lw=2, outer_lines=False):

    # If an axes object isn't provided to plot onto, just get current one

    if ax is None:

        ax = plt.gca()



    # Create the various parts of an NBA basketball court



    # Create the basketball hoop

    # Diameter of a hoop is 18" so it has a radius of 9", which is a value

    # 7.5 in our coordinate system

    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)



    # Create backboard

    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)



    # The paint

    # Create the outer box 0f the paint, width=16ft, height=19ft

    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color,

                          fill=False)

    # Create the inner box of the paint, widt=12ft, height=19ft

    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color,

                          fill=False)



    # Create free throw top arc

    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180,

                         linewidth=lw, color=color, fill=False)

    # Create free throw bottom arc

    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0,

                            linewidth=lw, color=color, linestyle='dashed')

    # Restricted Zone, it is an arc with 4ft radius from center of the hoop

    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw,

                     color=color)



    # Three point line

    # Create the side 3pt lines, they are 14ft long before they begin to arc

    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw,

                               color=color)

    corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)

    # 3pt arc - center of arc will be the hoop, arc is 23'9" away from hoop

    # I just played around with the theta values until they lined up with the 

    # threes

    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw,

                    color=color)



    # Center Court

    center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0,

                           linewidth=lw, color=color)

    center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0,

                           linewidth=lw, color=color)



    # List of the court elements to be plotted onto the axes

    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,

                      bottom_free_throw, restricted, corner_three_a,

                      corner_three_b, three_arc, center_outer_arc,

                      center_inner_arc]



    if outer_lines:

        # Draw the half court line, baseline and side out bound lines

        outer_lines = Rectangle((-250, -47.5), 500, 470, linewidth=lw,

                                color=color, fill=False)

        court_elements.append(outer_lines)



    # Add the court elements onto the axes

    for element in court_elements:

        ax.add_patch(element)



    return ax



def Draw2DGaussians(gaussianMixtureModel, ellipseColors, ellipseTextMessages):

    

    fig, h = plt.subplots();

    for i, (mean, covarianceMatrix) in enumerate(zip(gaussianMixtureModel.means_, gaussianMixtureModel.covariances_)):

        # get the eigen vectors and eigen values of the covariance matrix

        v, w = np.linalg.eigh(covarianceMatrix)

        v = 2.5*np.sqrt(v) # go to units of standard deviation instead of variance

        

        # calculate the ellipse angle and two axis length and draw it

        u = w[0] / np.linalg.norm(w[0])    

        angle = np.arctan(u[1] / u[0])

        angle = 180 * angle / np.pi  # convert to degrees

        currEllipse = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color=ellipseColors[i])

        currEllipse.set_alpha(0.5)

        h.add_artist(currEllipse)

        h.text(mean[0]+7, mean[1]-1, ellipseTextMessages[i], fontsize=13, color='blue')

#%% show gaussian mixture elipses of shot attempts

plt.rcParams['figure.figsize'] = (13, 10)

plt.rcParams['font.size'] = 15



ellipseTextMessages = [str(100*gaussianMixtureModel.weights_[x])[:4]+'%' for x in range(numGaussians)]

ellipseColors = ['red','green','purple','cyan','magenta','yellow','blue','orange','silver','maroon','lime','olive','brown','darkblue']

Draw2DGaussians(gaussianMixtureModel, ellipseColors, ellipseTextMessages)

draw_court(outer_lines=True); plt.ylim(-60,440); plt.xlim(270,-270); plt.title('shot attempts')
#%% just to make sure the gaussian model actually captures something, show the scatter and cluster assignment

plt.rcParams['figure.figsize'] = (13, 10)

plt.rcParams['font.size'] = 15



plt.figure(); draw_court(outer_lines=True); plt.ylim(-60,440); plt.xlim(270,-270); plt.title('cluser assignment')

plt.scatter(x=data['loc_x'],y=data['loc_y'],c=data['shotLocationCluster'],s=40,cmap='hsv',alpha=0.1)
#%% for each cluster, calculate it's individual accuracy and plot it

plt.rcParams['figure.figsize'] = (13, 10)

plt.rcParams['font.size'] = 15



variableCategories = data['shotLocationCluster'].value_counts().index.tolist()



clusterAccuracy = {}

for category in variableCategories:

    shotsAttempted = np.array(data['shotLocationCluster'] == category).sum()

    shotsMade = np.array(data.loc[data['shotLocationCluster'] == category,'shot_made_flag'] == 1).sum()

    clusterAccuracy[category] = float(shotsMade)/shotsAttempted



ellipseTextMessages = [str(100*clusterAccuracy[x])[:4]+'%' for x in range(numGaussians)]

Draw2DGaussians(gaussianMixtureModel, ellipseColors, ellipseTextMessages)

draw_court(outer_lines=True); plt.ylim(-60,440); plt.xlim(270,-270); plt.title('shot accuracy')
#%% plot a 2-d spatio-temporal histogram of kobe's games during his entire carrer

plt.rcParams['figure.figsize'] = (18, 10)

plt.rcParams['font.size'] = 18



# sort the clusters according to their accuracy

sortedClustersByAccuracyTuple = sorted(clusterAccuracy.items(), key=operator.itemgetter(1),reverse=True)

sortedClustersByAccuracy = [x[0] for x in sortedClustersByAccuracyTuple]



binSizeInSeconds = 12

timeInUnitsOfBins = ((data['secondsFromGameStart']+0.0001)/binSizeInSeconds).astype(int)

locationInUintsOfClusters = np.array([sortedClustersByAccuracy.index(data.loc[x,'shotLocationCluster']) for x in range(data.shape[0])])



# build a spatio-temporal histogram of Kobe's games

shotAttempts = np.zeros((gaussianMixtureModel.n_components,1+max(timeInUnitsOfBins)))

for shot in range(data.shape[0]):

    shotAttempts[locationInUintsOfClusters[shot],timeInUnitsOfBins[shot]] += 1



# make the y-axis have larger area so it will be more visible 

shotAttempts = np.kron(shotAttempts,np.ones((5,1)))

# the locations of the period ends

vlinesList = 0.5001+np.array([0,12*60,2*12*60,3*12*60,4*12*60,4*12*60+5*60]).astype(int)/binSizeInSeconds



plt.figure(figsize=(13,8))

plt.imshow(shotAttempts, cmap='copper',interpolation="nearest"); plt.xlim(0,float(4*12*60+6*60)/binSizeInSeconds);

plt.vlines(x=vlinesList, ymin=-0.5,ymax=shotAttempts.shape[0]-0.5, colors='r');

plt.xlabel('time from start of game [sec]'); plt.ylabel('cluster (sorted by accuracy)')
#%% create a new table for shot difficulty model

def FactorizeCategoricalVariable(inputDB,categoricalVarName):

    opponentCategories = inputDB[categoricalVarName].value_counts().index.tolist()

    

    outputDB = pd.DataFrame()

    for category in opponentCategories:

        featureName = categoricalVarName + ': ' + str(category)

        outputDB[featureName] = (inputDB[categoricalVarName] == category).astype(int)



    return outputDB



featuresDB = pd.DataFrame()

featuresDB['homeGame'] = data['matchup'].apply(lambda x: 1 if (x.find('@') < 0) else 0)

featuresDB = pd.concat([featuresDB,FactorizeCategoricalVariable(data,'opponent')],axis=1)

featuresDB = pd.concat([featuresDB,FactorizeCategoricalVariable(data,'action_type')],axis=1)

featuresDB = pd.concat([featuresDB,FactorizeCategoricalVariable(data,'shot_type')],axis=1)

featuresDB = pd.concat([featuresDB,FactorizeCategoricalVariable(data,'combined_shot_type')],axis=1)

featuresDB = pd.concat([featuresDB,FactorizeCategoricalVariable(data,'shot_zone_basic')],axis=1)

featuresDB = pd.concat([featuresDB,FactorizeCategoricalVariable(data,'shot_zone_area')],axis=1)

featuresDB = pd.concat([featuresDB,FactorizeCategoricalVariable(data,'shot_zone_range')],axis=1)

featuresDB = pd.concat([featuresDB,FactorizeCategoricalVariable(data,'shotLocationCluster')],axis=1)



featuresDB['playoffGame']          = data['playoffs']

featuresDB['locX']                 = data['loc_x']

featuresDB['locY']                 = data['loc_y']

featuresDB['distanceFromBasket']   = data['shot_distance']

featuresDB['secondsFromPeriodEnd'] = data['secondsFromPeriodEnd']



featuresDB['dayOfWeek_cycX']  = np.sin(2*np.pi*(data['dayOfWeek']/7))

featuresDB['dayOfWeek_cycY']  = np.cos(2*np.pi*(data['dayOfWeek']/7))

featuresDB['timeOfYear_cycX'] = np.sin(2*np.pi*(data['dayOfYear']/365))

featuresDB['timeOfYear_cycY'] = np.cos(2*np.pi*(data['dayOfYear']/365))



labelsDB = data['shot_made_flag']
#%% build a simple model and make sure it doesnt overfit

randomSeed = 1

numFolds   = 4



stratifiedCV = model_selection.StratifiedKFold(n_splits=numFolds, shuffle=True, random_state=randomSeed)



mainLearner = ensemble.ExtraTreesClassifier(n_estimators=500, max_depth=5, 

                                            min_samples_leaf=120, max_features=120, 

                                            criterion='entropy', bootstrap=False, 

                                            n_jobs=-1, random_state=randomSeed)



startTime = time.time()

trainAccuracy = []; validAccuracy = [];

trainLogLosses = []; validLogLosses = []

for trainInds, validInds in stratifiedCV.split(featuresDB, labelsDB):

    # split to train and valid sets

    X_train_CV = featuresDB.iloc[trainInds,:]

    y_train_CV = labelsDB.iloc[trainInds]

    X_valid_CV = featuresDB.iloc[validInds,:]

    y_valid_CV = labelsDB.iloc[validInds]

    

    # train learner

    mainLearner.fit(X_train_CV, y_train_CV)

    

    # make predictions

    y_train_hat_mainLearner = mainLearner.predict_proba(X_train_CV)[:,1]

    y_valid_hat_mainLearner = mainLearner.predict_proba(X_valid_CV)[:,1]



    # store results

    trainAccuracy.append(accuracy(y_train_CV, y_train_hat_mainLearner > 0.5))

    validAccuracy.append(accuracy(y_valid_CV, y_valid_hat_mainLearner > 0.5))

    trainLogLosses.append(log_loss(y_train_CV, y_train_hat_mainLearner))

    validLogLosses.append(log_loss(y_valid_CV, y_valid_hat_mainLearner))



print("-----------------------------------------------------")

print("total (train,valid) Accuracy = (%.5f,%.5f). took %.2f minutes" % (np.mean(trainAccuracy),np.mean(validAccuracy), (time.time()-startTime)/60))

print("total (train,valid) Log Loss = (%.5f,%.5f). took %.2f minutes" % (np.mean(trainLogLosses),np.mean(validLogLosses), (time.time()-startTime)/60))

print("-----------------------------------------------------")
mainLearner.fit(featuresDB, labelsDB)

data['shotDifficulty'] = mainLearner.predict_proba(featuresDB)[:,1]
# just to get a feel for what determins shot difficulty, look at feature importances

featureInds = mainLearner.feature_importances_.argsort()[::-1]

featureImportance = pd.DataFrame(np.concatenate((featuresDB.columns[featureInds,None], mainLearner.feature_importances_[featureInds,None]), axis=1),

                                  columns=['featureName', 'importanceET'])



featureImportance.iloc[:30,:]
#%% collect data given that kobe made or missed last shot

timeBetweenShotsDict = {}

timeBetweenShotsDict['madeLast']   = []

timeBetweenShotsDict['missedLast'] = []



changeInDistFromBasketDict = {}

changeInDistFromBasketDict['madeLast']   = []

changeInDistFromBasketDict['missedLast'] = []



changeInShotDifficultyDict = {}

changeInShotDifficultyDict['madeLast']   = []

changeInShotDifficultyDict['missedLast'] = []



afterMadeShotsList   = []

afterMissedShotsList = []



for shot in range(1,data.shape[0]):



    # make sure the current shot and last shot were all in the same period of the same game

    sameGame   = data.loc[shot,'game_date'] == data.loc[shot-1,'game_date']

    samePeriod = data.loc[shot,'period']    == data.loc[shot-1,'period']



    if samePeriod and sameGame:

        madeLastShot       = data.loc[shot-1,'shot_made_flag'] == 1

        missedLastShot     = data.loc[shot-1,'shot_made_flag'] == 0

        

        timeDifferenceFromLastShot = data.loc[shot,'secondsFromGameStart']     - data.loc[shot-1,'secondsFromGameStart']

        distDifferenceFromLastShot = data.loc[shot,'shot_distance']            - data.loc[shot-1,'shot_distance']

        shotDifficultyDifferenceFromLastShot = data.loc[shot,'shotDifficulty'] - data.loc[shot-1,'shotDifficulty']



        # check for currupt data points (assuming all samples should have been chronologically ordered)

        if timeDifferenceFromLastShot < 0:

            continue

        

        if madeLastShot:

            timeBetweenShotsDict['madeLast'].append(timeDifferenceFromLastShot)

            changeInDistFromBasketDict['madeLast'].append(distDifferenceFromLastShot)

            changeInShotDifficultyDict['madeLast'].append(shotDifficultyDifferenceFromLastShot)

            afterMadeShotsList.append(shot)

            

        if missedLastShot:

            timeBetweenShotsDict['missedLast'].append(timeDifferenceFromLastShot)

            changeInDistFromBasketDict['missedLast'].append(distDifferenceFromLastShot)

            changeInShotDifficultyDict['missedLast'].append(shotDifficultyDifferenceFromLastShot)

            afterMissedShotsList.append(shot)



afterMissedData = data.iloc[afterMissedShotsList,:]

afterMadeData   = data.iloc[afterMadeShotsList,:]



shotChancesListAfterMade = afterMadeData['shotDifficulty'].tolist()

totalAttemptsAfterMade   = afterMadeData.shape[0]

totalMadeAfterMade       = np.array(afterMadeData['shot_made_flag'] == 1).sum()



shotChancesListAfterMissed = afterMissedData['shotDifficulty'].tolist()

totalAttemptsAfterMissed   = afterMissedData.shape[0]

totalMadeAfterMissed       = np.array(afterMissedData['shot_made_flag'] == 1).sum()
#%% after making a shot, kobe wants more

plt.rcParams['figure.figsize'] = (13, 10)



jointHist, timeBins = np.histogram(timeBetweenShotsDict['madeLast']+timeBetweenShotsDict['missedLast'],bins=200)

barWidth = 0.999*(timeBins[1]-timeBins[0])



timeDiffHist_GivenMadeLastShot, b = np.histogram(timeBetweenShotsDict['madeLast'],bins=timeBins)

timeDiffHist_GivenMissedLastShot, b = np.histogram(timeBetweenShotsDict['missedLast'],bins=timeBins)

maxHeight = max(max(timeDiffHist_GivenMadeLastShot),max(timeDiffHist_GivenMissedLastShot)) + 30



plt.figure();

plt.subplot(2,1,1); plt.bar(timeBins[:-1], timeDiffHist_GivenMadeLastShot, width=barWidth); plt.xlim((0,500)); plt.ylim((0,maxHeight))

plt.title('made last shot'); plt.ylabel('counts')

plt.subplot(2,1,2); plt.bar(timeBins[:-1], timeDiffHist_GivenMissedLastShot, width=barWidth); plt.xlim((0,500)); plt.ylim((0,maxHeight))

plt.title('missed last shot'); plt.xlabel('time since last shot'); plt.ylabel('counts')
#%% to make the difference clearer, show the cumulative histogram

plt.rcParams['figure.figsize'] = (13, 6)



timeDiffCumHist_GivenMadeLastShot = np.cumsum(timeDiffHist_GivenMadeLastShot).astype(float)

timeDiffCumHist_GivenMadeLastShot = timeDiffCumHist_GivenMadeLastShot/max(timeDiffCumHist_GivenMadeLastShot)

timeDiffCumHist_GivenMissedLastShot = np.cumsum(timeDiffHist_GivenMissedLastShot).astype(float)

timeDiffCumHist_GivenMissedLastShot = timeDiffCumHist_GivenMissedLastShot/max(timeDiffCumHist_GivenMissedLastShot)



maxHeight = max(timeDiffCumHist_GivenMadeLastShot[-1],timeDiffCumHist_GivenMissedLastShot[-1])



plt.figure();

madePrev = plt.plot(timeBins[:-1], timeDiffCumHist_GivenMadeLastShot, label='made Prev'); plt.xlim((0,500))

missedPrev = plt.plot(timeBins[:-1], timeDiffCumHist_GivenMissedLastShot, label='missed Prev'); plt.xlim((0,500)); plt.ylim((0,1))

plt.title('cumulative density function - CDF'); plt.xlabel('time since last shot'); plt.legend(loc='lower right')
#%% after making a shot, kobe is a more confident and throws from further away

plt.rcParams['figure.figsize'] = (13, 10)



jointHist, distDiffBins = np.histogram(changeInDistFromBasketDict['madeLast']+changeInDistFromBasketDict['missedLast'],bins=100,density=False)

barWidth = 0.999*(distDiffBins[1]-distDiffBins[0])



distDiffHist_GivenMadeLastShot,   b = np.histogram(changeInDistFromBasketDict['madeLast'],bins=distDiffBins)

distDiffHist_GivenMissedLastShot, b = np.histogram(changeInDistFromBasketDict['missedLast'],bins=distDiffBins)

maxHeight = max(max(distDiffHist_GivenMadeLastShot),max(distDiffHist_GivenMissedLastShot)) + 30



plt.figure();

plt.subplot(2,1,1); plt.bar(distDiffBins[:-1], distDiffHist_GivenMadeLastShot, width=barWidth); plt.xlim((-40,40)); plt.ylim((0,maxHeight))

plt.title('made last shot'); plt.ylabel('counts')

plt.subplot(2,1,2); plt.bar(distDiffBins[:-1], distDiffHist_GivenMissedLastShot, width=barWidth); plt.xlim((-40,40)); plt.ylim((0,maxHeight))

plt.title('missed last shot'); plt.xlabel('curr shot distance - prev shot distance'); plt.ylabel('counts')
#%% to make the difference clearer, show the cumulative histogram

plt.rcParams['figure.figsize'] = (13, 6)



distDiffCumHist_GivenMadeLastShot = np.cumsum(distDiffHist_GivenMadeLastShot).astype(float)

distDiffCumHist_GivenMadeLastShot = distDiffCumHist_GivenMadeLastShot/max(distDiffCumHist_GivenMadeLastShot)

distDiffCumHist_GivenMissedLastShot = np.cumsum(distDiffHist_GivenMissedLastShot).astype(float)

distDiffCumHist_GivenMissedLastShot = distDiffCumHist_GivenMissedLastShot/max(distDiffCumHist_GivenMissedLastShot)



maxHeight = max(distDiffCumHist_GivenMadeLastShot[-1],distDiffCumHist_GivenMissedLastShot[-1])



plt.figure();

madePrev = plt.plot(distDiffBins[:-1], distDiffCumHist_GivenMadeLastShot, label='made Prev'); plt.xlim((-40,40))

missedPrev = plt.plot(distDiffBins[:-1], distDiffCumHist_GivenMissedLastShot, label='missed Prev'); plt.xlim((-40,40)); plt.ylim((0,1))

plt.title('cumulative density function - CDF'); plt.xlabel('curr shot distance - prev shot distance'); plt.legend(loc='lower right')
#%% after making a shot, kobe is a more confident and makes much more difficult shots generally

plt.rcParams['figure.figsize'] = (13, 10)



jointHist, difficultyDiffBins = np.histogram(changeInShotDifficultyDict['madeLast']+changeInShotDifficultyDict['missedLast'],bins=100)

barWidth = 0.999*(difficultyDiffBins[1]-difficultyDiffBins[0])



shotDifficultyDiffHist_GivenMadeLastShot,   b = np.histogram(changeInShotDifficultyDict['madeLast'],bins=difficultyDiffBins)

shotDifficultyDiffHist_GivenMissedLastShot, b = np.histogram(changeInShotDifficultyDict['missedLast'],bins=difficultyDiffBins)

maxHeight = max(max(shotDifficultyDiffHist_GivenMadeLastShot),max(shotDifficultyDiffHist_GivenMissedLastShot)) + 30



plt.figure();

plt.subplot(2,1,1); plt.bar(difficultyDiffBins[:-1], shotDifficultyDiffHist_GivenMadeLastShot, width=barWidth); plt.xlim((-1,1)); plt.ylim((0,maxHeight))

plt.title('made last shot'); plt.ylabel('counts')

plt.subplot(2,1,2); plt.bar(difficultyDiffBins[:-1], shotDifficultyDiffHist_GivenMissedLastShot, width=barWidth); plt.xlim((-1,1)); plt.ylim((0,maxHeight))

plt.title('missed last shot'); plt.xlabel('chance to make curr shot - chance to make prev shot'); plt.ylabel('counts')
#%% is this regression to the mean?

plt.rcParams['figure.figsize'] = (12, 10)



accuracyAllShots    = data['shot_made_flag'].mean()

accuracyAfterMade   = afterMadeData['shot_made_flag'].mean()

accuracyAfterMissed = afterMissedData['shot_made_flag'].mean()



standardErrorAllShots    = np.sqrt(accuracyAllShots*(1-accuracyAllShots)/data.shape[0])

standardErrorAfterMade   = np.sqrt(accuracyAfterMade*(1-accuracyAfterMade)/afterMadeData.shape[0])

standardErrorAfterMissed = np.sqrt(accuracyAfterMissed*(1-accuracyAfterMissed)/afterMissedData.shape[0])



accuracyVec = np.array([accuracyAfterMade,accuracyAllShots,accuracyAfterMissed])

errorVec    = np.array([standardErrorAfterMade,standardErrorAllShots,standardErrorAfterMissed])



barWidth = 0.7

xLocs = np.arange(len(accuracyVec)) + 0.5



fig, h = plt.subplots(); h.bar(xLocs, accuracyVec, barWidth, color='b', yerr=errorVec)

h.set_xticks(xLocs); h.set_xticklabels(('after made', 'all shots', 'after missed'))

plt.ylim([0.41,0.47]); plt.xlim([-0.3,3.3]); plt.title('not regression to the mean')
#%% but wait, maybe kobe is making more difficult shots because he's "in the zone"



predictedShotPercentAfterMade = np.array(shotChancesListAfterMade).mean()

predictedStadardDev = np.sqrt(predictedShotPercentAfterMade*(1-predictedShotPercentAfterMade))

stadardError = predictedStadardDev/np.sqrt(len(shotChancesListAfterMade))

predPlusErr  = predictedShotPercentAfterMade + 2*stadardError

predMinusErr = predictedShotPercentAfterMade - 2*stadardError

actualShotPercentAfterMade = float(totalMadeAfterMade)/totalAttemptsAfterMade



print("-----------------------------------------------------")

print('provided that kobe MADE the previous shot:')

print('according to "shotDifficulty" model, 95% confidence interval ['+ str(predMinusErr)+', '+str(predPlusErr)+']')

print('and Kobe actually made ' + str(actualShotPercentAfterMade) + ', which is within confidence interval')

print("-----------------------------------------------------")



predictedShotPercentAfterMissed = np.array(shotChancesListAfterMissed).mean()

predictedStadardDev = np.sqrt(predictedShotPercentAfterMissed*(1-predictedShotPercentAfterMissed))

stadardError = predictedStadardDev/np.sqrt(len(shotChancesListAfterMissed))

predPlusErr  = predictedShotPercentAfterMissed + 2*stadardError

predMinusErr = predictedShotPercentAfterMissed - 2*stadardError

actualShotPercentAfterMissed = float(totalMadeAfterMissed)/totalAttemptsAfterMissed



print("-----------------------------------------------------")

print('provided that kobe MISSED the previous shot:')

print('according to "shotDifficulty" model, 95% confidence interval ['+ str(predMinusErr)+', '+str(predPlusErr)+']')

print('and Kobe actually made ' + str(actualShotPercentAfterMissed) + ', which is within confidence interval')

print("-----------------------------------------------------")
#%% let's try and visualize this - show scatter plot of after made and after missed shots

plt.rcParams['figure.figsize'] = (16, 8)



afterMissedData = data.iloc[afterMissedShotsList,:]

afterMadeData = data.iloc[afterMadeShotsList,:]



plt.figure();

plt.subplot(1,2,1); plt.title('shots after made')

plt.scatter(x=afterMadeData['loc_x'],y=afterMadeData['loc_y'],c=afterMadeData['shotLocationCluster'],s=50,cmap='hsv',alpha=0.06)

draw_court(outer_lines=True); plt.ylim(-60,440); plt.xlim(270,-270);



plt.subplot(1,2,2); plt.title('shots after missed');

plt.scatter(x=afterMissedData['loc_x'],y=afterMissedData['loc_y'],c=afterMissedData['shotLocationCluster'],s=50,cmap='hsv',alpha=0.06)

draw_court(outer_lines=True); plt.ylim(-60,440); plt.xlim(270,-270);
#%% show shot attempts of after made and after missed shots

plt.rcParams['figure.figsize'] = (13, 10)



variableCategories = afterMadeData['shotLocationCluster'].value_counts().index.tolist()

clusterFrequency = {}

for category in variableCategories:

    shotsAttempted = np.array(afterMadeData['shotLocationCluster'] == category).sum()

    clusterFrequency[category] = float(shotsAttempted)/afterMadeData.shape[0]



ellipseTextMessages = [str(100*clusterFrequency[x])[:4]+'%' for x in range(numGaussians)]

Draw2DGaussians(gaussianMixtureModel, ellipseColors, ellipseTextMessages)

draw_court(outer_lines=True); plt.ylim(-60,440); plt.xlim(270,-270); plt.title('after made shots')



variableCategories = afterMissedData['shotLocationCluster'].value_counts().index.tolist()

clusterFrequency = {}

for category in variableCategories:

    shotsAttempted = np.array(afterMissedData['shotLocationCluster'] == category).sum()

    clusterFrequency[category] = float(shotsAttempted)/afterMissedData.shape[0]



ellipseTextMessages = [str(100*clusterFrequency[x])[:4]+'%' for x in range(numGaussians)]

Draw2DGaussians(gaussianMixtureModel, ellipseColors, ellipseTextMessages)

draw_court(outer_lines=True); plt.ylim(-60,440); plt.xlim(270,-270); plt.title('after missed shots')