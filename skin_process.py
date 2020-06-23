import pandas as pd
import numpy as np
from pyod.models.cblof import CBLOF
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
import os
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import random
path = 'data/data/skin/benchmarks/'
path_new='data/data/skin/results/'
files = os.listdir(path)
train_csv = list(files)
label=[['clf_name','out_rate','R_out_count','P_out_count','P_nor_count','AUC','AP']]
label_D=pd.DataFrame(label)
label_D.to_csv(path_new+"CBLOF_re.csv",mode='a',index=False, header=False)
label_D.to_csv(path_new+"KNN_re.csv",mode='a',index=False, header=False)
label_D.to_csv(path_new+"Isolation Forest_re.csv",mode='a',index=False, header=False)
label_D.to_csv(path_new+"Feature Bagging_re.csv",mode='a',index=False, header=False)
label_D.to_csv(path_new+"HBOS_re.csv",mode='a',index=False, header=False)
for i in range(len(train_csv)):
    print("正在处理的文件为：%s" %(train_csv[i]))
    df = pd.read_csv(path+train_csv[i])
    label=[]
    X=df[['R','G','B']]
    count=0
    for i in range(len(df)):
        if df['ground.truth'][i]=="nominal":
            label+=[1]
        if df['ground.truth'][i]=='anomaly':
            label+=[0]
            count+=1
    Y=label
    random_state = np.random.RandomState(42)
    outliers_rate = count/len(df)
    outliers_fraction=outliers_rate
    if outliers_fraction>=0.5:
        outliers_fraction=0.4
    classifiers = {
        'CBLOF':CBLOF(contamination=outliers_fraction,check_estimator=False, random_state=random_state),
        'KNN': KNN(contamination=outliers_fraction),
        'HBOS': HBOS(contamination=outliers_fraction),
        'Isolation Forest': IForest(contamination=outliers_fraction,random_state=random_state),
        'Feature Bagging':FeatureBagging(LOF(n_neighbors=35),contamination=outliers_fraction,check_estimator=False,random_state=random_state),
    }
    print("算法运行中：")
    for i, (clf_name, clf) in enumerate(classifiers.items()):
        print("算法：%s 正在运行" % (clf_name))
        clf.fit(X)
        y_pred=clf.labels_ 
        n_out=np.count_nonzero(y_pred)
        n_normal=len(y_pred)-n_out
        print("原数据的异常值数量为：%d;异常值数量：%d;正常值：%d" %(count,n_out,n_normal))
        
        for i in range(len(y_pred)):
            if(y_pred[i]==1):
                y_pred[i]=0
            else:
                y_pred[i]=1
        AUC=roc_auc_score(Y, y_pred)
        AP=average_precision_score(Y, y_pred)
        list=[[clf_name,outliers_rate,count,n_out,n_normal,AUC,AP]]
        list_d=pd.DataFrame(list)
        list_d.to_csv(path_new+clf_name+"_re.csv",mode='a',index=False, header=False)
        print("AUC=%4lf;AP=%4lf" %(AUC,AP))
        print("算法：%s 运行完毕" % (clf_name))