
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize

main = tkinter.Tk()
main.title("PREDICTION OF LOAN ELIGIBILITY OF THE CUSTOMER") #designing main screen
main.geometry("1300x1200")

global filename
precision = []
recall = []
fscore = []
accuracy = []
global X, Y
global dataset
global X_train, X_test, y_train, y_test
global le
global classifier

def upload(): #function to upload tweeter profile
    global filename
    global dataset
    filename = filedialog.askopenfilename(initialdir="Dataset")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n");
    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace = True)
    text.insert(END,str(dataset.head()))
    dataset['Loan_ID'] = dataset['Loan_ID'].astype('str')
    dataset['Gender'] = dataset['Gender'].astype('str')
    dataset['Married'] = dataset['Married'].astype('str')
    dataset['Education'] = dataset['Education'].astype('str')
    dataset['Self_Employed'] = dataset['Self_Employed'].astype('str')
    dataset['Property_Area'] = dataset['Property_Area'].astype('str')
    dataset['Loan_Status'] = dataset['Loan_Status'].astype('str')
    print(dataset.info())
    sns.set_style('dark')
    dataset.plot(figsize=(18, 8))
    plt.show()

def preprocess():
    global dataset
    global le
    text.delete('1.0', END)
    le = LabelEncoder()
    dataset.drop(['Loan_ID'], axis = 1,inplace=True)
    dataset['Gender'] = pd.Series(le.fit_transform(dataset['Gender']))
    dataset['Married'] = pd.Series(le.fit_transform(dataset['Married']))
    dataset['Education'] = pd.Series(le.fit_transform(dataset['Education']))
    dataset['Self_Employed'] = pd.Series(le.fit_transform(dataset['Self_Employed']))
    dataset['Property_Area'] = pd.Series(le.fit_transform(dataset['Property_Area']))
    dataset['Loan_Status'] = pd.Series(le.fit_transform(dataset['Loan_Status']))
    text.insert(END,str(dataset.head()))
    

def splitDataset():
    text.delete('1.0', END)
    global filename
    global dataset
    global X, Y
    global X_train, X_test, y_train, y_test
    dataset = dataset.values
    cols = dataset.shape[1]-1
    X = dataset[:,0:cols]
    Y = dataset[:,cols]
    X = normalize(X)
    print(Y)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Total records found in dataset are : "+str(X.shape[0])+"\n")
    text.insert(END,"Total records used to train machine learning algorithms are : "+str(X_train.shape[0])+"\n")
    text.insert(END,"Total records used to test machine learning algorithms are  : "+str(X_test.shape[0])+"\n")
    dataset = pd.read_csv(filename)
    plt.figure(figsize=(75,75))
    sns.heatmap(dataset.corr(), annot = True)
    plt.show()
                
def runRF():
    global classifier
    precision.clear()
    recall.clear()
    fscore.clear()
    accuracy.clear()
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test
    cls = RandomForestClassifier(n_estimators=200,random_state=0)
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test) 
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = accuracy_score(y_test,predict)*100
    text.insert(END,'Random Forest Accuracy  : '+str(a)+"\n")
    text.insert(END,'Random Forest Precision : '+str(p)+"\n")
    text.insert(END,'Random Forest Recall    : '+str(r)+"\n")
    text.insert(END,'Random Forest FSCORE    : '+str(f)+"\n\n")
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    classifier = cls

def predictEligibility():
    global classifier
    global le
    text.delete('1.0', END)
    testname = filedialog.askopenfilename(initialdir = "Dataset")
    test = pd.read_csv(testname)
    test.fillna(0, inplace = True)
    test['Loan_ID'] = test['Loan_ID'].astype('str')
    test['Gender'] = test['Gender'].astype('str')
    test['Married'] = test['Married'].astype('str')
    test['Education'] = test['Education'].astype('str')
    test['Self_Employed'] = test['Self_Employed'].astype('str')
    test['Property_Area'] = test['Property_Area'].astype('str')
    test.drop(['Loan_ID'], axis = 1,inplace=True)
    test['Gender'] = pd.Series(le.fit_transform(test['Gender']))
    test['Married'] = pd.Series(le.fit_transform(test['Married']))
    test['Education'] = pd.Series(le.fit_transform(test['Education']))
    test['Self_Employed'] = pd.Series(le.fit_transform(test['Self_Employed']))
    test['Property_Area'] = pd.Series(le.fit_transform(test['Property_Area']))
    test = test.values
    test = normalize(test)
    cols = test.shape[1]
    test = test[:,0:cols]
    predict = classifier.predict(test)
    print(predict)
    for i in range(len(predict)):
        if predict[i] == 0:
            text.insert(END,"Test Record : "+str(test[i])+" Sorry! Not Eligible for Loan\n\n")
        else:
            text.insert(END,"Test Record : "+str(test[i])+" Congratulation! You are Eligible for Loan\n\n")
    
def graph():
    df = pd.DataFrame([['Random Forest','Precision',precision[0]],['Random Forest','Recall',recall[0]],['Random Forest','F1 Score',fscore[0]],['Random Forest','Accuracy',accuracy[0]],
                       
                      ],columns=['Parameters','Algorithm','Value'])
    df.pivot("Parameters", "Algorithm", "Value").plot(kind='bar')
    plt.show()


def close():
    main.destroy()

font = ('times', 16, 'bold')
title = Label(main, text='PREDICTION OF LOAN ELIGIBILITY OF THE CUSTOMER')
title.config(bg='brown', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Loan Dataset", command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=360,y=100)

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocess)
preprocessButton.place(x=50,y=150)
preprocessButton.config(font=font1) 

traintestButton = Button(main, text="Generate Train & Test Data", command=splitDataset)
traintestButton.place(x=340,y=150)
traintestButton.config(font=font1) 

rfButton = Button(main, text="Run Random Forest ML Model", command=runRF)
rfButton.place(x=630,y=150)
rfButton.config(font=font1) 

predictButton = Button(main, text="Predict Eligibility using RF Model", command=predictEligibility)
predictButton.place(x=920,y=150)
predictButton.config(font=font1)

#graphButton = Button(main, text="Random Forest Performance Graph", command=graph)
#graphButton.place(x=50,y=200)
#graphButton.config(font=font1)

closeButton = Button(main, text="Exit", command=close)
closeButton.place(x=340,y=200)
closeButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=18,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1)


main.config(bg='brown')
main.mainloop()
