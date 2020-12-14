import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="decision_tree",
	help="type of python machine learning model to use")
args = vars(ap.parse_args())

models = {
	"decision_tree": DecisionTreeClassifier(criterion = "entropy", random_state=84, max_depth=6, min_samples_leaf=3),
	"naive_bayes": GaussianNB()
}

df = pd.read_csv("diabetes_data_upload.csv",delimiter=",")
print(len(df))
print(df.shape)
print(df.info())
df.iloc[:,1:17]=df.iloc[:,1:17].apply(LabelEncoder().fit_transform)
data =df.iloc[:,0:16]
data1 =df.iloc[:,[2,3,4,5,8,9,10,13]]
targe =df['class']
targe1 =df['class']
total_acc =0
total_acc1 =0
print("[INFO] using '{}' model".format(args["model"]))
model = models[args["model"]]
for i in range(0, 10):
	x_train,x_test,y_train,y_test  = train_test_split(data,targe,shuffle=True,test_size=0.25,random_state = i)



	print("So luong du lieu hoc ",len(x_train))
	print("So luong du lieu kiem tra ",len(x_test))
	model.fit(x_train,y_train)

	dubao = model.predict(x_test)

	print ("Do chinh xac tong the lan lap thu", i ," cua tat tap du lieu la : ", accuracy_score(y_test,dubao)*100)

	total_acc +=accuracy_score(y_test,dubao)

	lb = np.unique(dubao)

	cnf_matrix_gnb = confusion_matrix(y_test, dubao,labels=lb)

	print(cnf_matrix_gnb)

		
print("do chinh xac tong the ",total_acc/10)			
B=np.array([[40,1,0,1,1,1,0,1,1,1,0,1,0,0,0,0],
  	[50,0,1,1,0,0,0,1,0,1,0,1,0,1,1,0],
  	[20,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
print(model.predict(B))
print("[INFO] using '{}' model".format(args["model"]))
model1 = models[args["model"]]
for i in range(0, 10):
	x_train1,x_test1,y_train1,y_test1  = train_test_split(data1,targe1,test_size=0.25,random_state = i)

	model1.fit(x_train1,y_train1)
	dubao1 = model1.predict(x_test1)	
	print ("Do chinh xac tong the lan lap thu", i ," cua du lieu duoc chon la : ", accuracy_score(y_test1,dubao1)*100)
	total_acc1 +=accuracy_score(y_test1,dubao1)
	lb1= np.unique(dubao1)
	cnf_matrix_gnb1 = confusion_matrix(y_test1, dubao1,labels=lb1)	
	print(cnf_matrix_gnb1)
	
print("do chinh xac tong the ",total_acc1/10)	
B=np.array([[0,1,1,1,1,1,0,0],
  	[1,1,0,0,0,1,0,1],
  	[0,0,0,0,0,0,0,0]])
print(model.predict(B))	
