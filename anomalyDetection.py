# -*- coding: utf-8 -*-
"""
Created on Tue Jan 02 21:22:35 2018

@author: Volta
"""

from __future__ import division
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn import svm
import matplotlib.font_manager
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

df0 = pd.read_excel('C:\\inputs.xlsx')
currentSeries=5
colName='TS'+str(currentSeries)

#=======================================================
# №0 Find using plot visually ==========================
#=======================================================
df=df0.loc[:,['date',colName]].dropna()
df.plot(x='date',y=colName)

#=======================================================
# №1 Find using Boxplot (+/- 1.5 IQR) ==================
#=======================================================
df=df0.iloc[:,currentSeries].dropna()
plt.figure()
plt.boxplot(df)
plt.ion()
plt.show()

firstQuantile=np.percentile(df, 25)
thirdQuantile=np.percentile(df, 75)
IQR=thirdQuantile-firstQuantile
minBorder=firstQuantile-1.5*IQR
maxBorder=thirdQuantile+1.5*IQR

smallOutliers=df.where(df<minBorder).dropna()
bigOutliers=df.where(df>maxBorder).dropna()
print("Method 1 (1.5 IQR): Outliers with the smallest values:",smallOutliers)
print("Method 1 (1.5 IQR): Outliers with the biggest values:",bigOutliers)

#=======================================================
# №2 Find using 3 Standard Deviations ==================
#=======================================================
df=df0.iloc[:,currentSeries].dropna()
sigma=np.std(df)
avg=np.average(df)
smallestOutliers=df.where(df<avg-3*sigma).dropna()
biggestOutliers=df.where(df>avg+3*sigma).dropna()
print("Method 2 (3 Std Devs): Outliers with the smallest values:",smallestOutliers)
print("Method 2 (3 Std Devs): Outliers with the biggest values:",biggestOutliers)

#=======================================================
# №3 Find using 3 One-class Support Vector Machine =====
#=======================================================
df=df0.loc[:,['date',colName]].dropna()

df=df0.dropna()

df_params = np.array(df.values[:,2:], dtype="float64")
df_params = scale(df_params)


X = PCA(n_components=2).fit_transform(df_params)
df_num = X.shape[0]
OUTLIER_FRACTION = 0.01

clf = svm.OneClassSVM(kernel="rbf")
clf.fit(X)

dist_to_border = clf.decision_function(X).ravel()
threshold = stats.scoreatpercentile(dist_to_border,
            100 * OUTLIER_FRACTION)
is_inlier = dist_to_border > threshold

xx, yy = np.meshgrid(np.linspace(-7, 7, 500), np.linspace(-7,7, 500))
n_inliers = int((1. - OUTLIER_FRACTION) * df_num)
n_outliers = int(OUTLIER_FRACTION * df_num)
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.title("Outlier detection")
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),
                         cmap=plt.cm.Blues_r)
a = plt.contour(xx, yy, Z, levels=[threshold],
                            linewidths=2, colors='orange')
plt.contourf(xx, yy, Z, levels=[threshold, Z.max()],
                         colors='yellow')
b = plt.scatter(X[is_inlier == 0, 0], X[is_inlier == 0, 1], c='red')
c = plt.scatter(X[is_inlier == 1, 0], X[is_inlier == 1, 1], c='black')
plt.axis('tight')
plt.legend([a.collections[0], b, c],
           ['learned decision function', 'outliers', 'inliers'],
           prop=matplotlib.font_manager.FontProperties(size=11))
plt.xlim((-7, 7))
plt.ylim((-7, 7))
plt.show()
print df[is_inlier == 0]




