#!/usr/bin/env python
# coding: utf-8

# In[124]:


#------------------------------------------------------
#            Start               
#------------------------------------------------------


# In[125]:


#------------------------------------------------------
#             Import Libary                  
#------------------------------------------------------
import matplotlib.pyplot as plt
import numpy  as np
import pandas as pd
import time   as ti
import statsmodels.api as sm
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy
#------------------------------------------------------


# In[126]:


#------------------------------------------------------
#                    Define D
#------------------------------------------------------
def Matrix_diag(K,W):
    Matrix=np.matrix(W)
    S=np.dot(W.T,W)
    S_List=[]
    for i in range(K):
        S_List.append(S[i][i])
    diag=np.diag(S_List)
    D=(diag**(1/2))
    return D
#------------------------------------------------------


# In[127]:


#------------------------------------------------------
# Define For Select Row Of Matrix                    
#------------------------------------------------------
def Index_Select(matrix, i):
    return [row[i] for row in matrix]
#------------------------------------------------------


# In[128]:


#------------------------------------------------------
#             Definision MFFS                  
#------------------------------------------------------
def MFFS(X,K,Steps,Alpha,Break_Value):

    #-----------
    #Inter Parametr And Need Value
    #-----------
    X                   =  np.array(Main_Data_Set)
    Y                   =  np.array(Main_Data_Set)
    Row_Number          =  len(X)
    Column_Number       =  len(X[0])
    W                   =  np.random.rand(Column_Number,K)
    H                   =  np.random.rand(K,Column_Number)
    W_Final_Norm        =  []
    W_Final_Norm_Index  =  []
    F_List              =  []
    F_Steps_List        =  []
    #-----------
    #End
    #-----------
    
    
    #-----------  
    #Start Algoritm
    #-----------
    Start=ti.time()
    for i in range(Steps):
        
        
        #----------- 
        D=Matrix_diag(K,W)
        #----------- 
        
        #----------- 
        W=np.dot(W,np.linalg.inv(D))
        #----------- 
        
        #----------- 
        H=np.dot(D,H)
        #----------- 

        
        
        
        #----------- 
        W_UP=np.dot(np.dot(X.T,X),H.T) + Alpha*W
        W_DOWN=np.dot(np.dot(np.dot(np.dot(X.T,X),W),H),H.T) + Alpha*(np.dot(np.dot(W,W.T),W))
        W=W*(W_UP/W_DOWN)
        #----------- 
        
        
        #----------- 
        H_UP=(np.dot(np.dot(W.T,X.T),X))
        H_DOWN=np.dot(np.dot(np.dot(np.dot(W.T,X.T),X),W),H)
        H=H*(H_UP/H_DOWN)
        #----------- 
        
        #----------- 
        Error=np.linalg.norm(X-(np.dot(np.dot(X,W),H)))
        #----------- 
        
        
        #-----------
        F_Part_1= (1/2)*((np.linalg.norm(X-np.dot(np.dot(X,W),H)))**2)
        F_Part_2=(Alpha/4)*((np.linalg.norm(np.dot(W,W.T)-Error))**2)
        F=F_Part_1+F_Part_2
        F_List.append(F)
        F_Steps_List.append(i)
        #-----------
        
        
        
        #-----------
        if Error<=Break_Value:
            break
        #-----------
        
    #-----------
    #End Algoritm
    #-----------  
    

    
    
    #-----------
    #Calculate Norm OF W And Sort Index And Norm
    #-----------
    for i in range(0,Column_Number):
        W_Norm          = np.linalg.norm(W[i])
        W_Final_Norm.append(W_Norm)
        W_Final_Norm_Index.append(i+1)
        
        
    W_Final_Norm        = np.array(W_Final_Norm)
    W_Final_Norm_Index  = np.array(W_Final_Norm_Index)
    W_Norm_Index        = np.array([[W_Final_Norm],[W_Final_Norm_Index]])
    W_Norm_Index        = W_Norm_Index.T
    W_Norm_Index        = np.matrix(W_Norm_Index)
    W_Sorted            = W_Norm_Index[np.argsort(W_Norm_Index.A[:, 0])]
    W_Sorted            = np.array(W_Sorted)
    Final_Index         = Index_Select(W_Sorted,1)
    Final_Index.reverse()
    W_Sorted            = np.array(W_Sorted)
    Final_Norm          = Index_Select(W_Sorted,0)
    Final_Norm.reverse()
    #-----------  
    #End
    #----------- 
    
    
    
    #-----------  
    #Select Need Dimension Of Main Dataset  And Show    
    #-----------
    
    Dimension_Index_List=[]
    Final_Index_List=[]
    for i in range(K):
        My_Index=int(Final_Index[i]-1)
        Dimension_Index_List.append(My_Index)
        Final_Index_List.append(int(Final_Index[i]))
    Data_Set_Main=np.array(Y.T)
    Selected_Column=Data_Set_Main[Dimension_Index_List]
    return Selected_Column.T
    #-----------  
    #End  
    #-----------
#------------------------------------------------------


# In[129]:


#------------------------------------------------------
#             Definishion K_Means Algoritm (Clustering)                 
#------------------------------------------------------
def K_Means_Clustering(Data_Set,Count_Of_Cluster):
    
    kmeans = KMeans(Count_Of_Cluster)
    kmeans.fit(Data_Set)
    identified_clusters = kmeans.fit_predict(Data_Set)
    kmeans=pd.DataFrame(identified_clusters, index= None)
    kmeans_Label=np.matrix(kmeans)
    return kmeans_Label
#------------------------------------------------------


# In[130]:


#------------------------------------------------------
#             Definishion accuracy                 
#------------------------------------------------------
def Acc(Main_Labels,K_Labels):
    
    P=Main_Labels
    Q=np.array(K_Labels).ravel().tolist()
    
    return accuracy_score(P,Q)
#------------------------------------------------------


# In[131]:


#------------------------------------------------------
#             Definishion normalized mutualinformation              
#------------------------------------------------------
def NMI(Main_Labels,K_Labels):
    
    P=Main_Labels
    Q=np.array(K_Labels).ravel().tolist()
    I_PQ=mutual_info_score(P,Q)
    H_P=entropy(P)
    H_Q=entropy(Q)
    
    return I_PQ/((H_P*H_Q)**(1/2))
#------------------------------------------------------


# In[153]:


#------------------------------------------------------
#             Calculate Count Of Cluster          
#------------------------------------------------------
def Cluster_Count(Main_Labels):
    
    input_list=Main_Labels
    l1 = []
    count = 0
    for item in input_list:
        if item not in l1:
            count += 1
            l1.append(item)
    Cluster_Count=count
    return Cluster_Count


# In[213]:


def Multi_MFFS(K):
    Data_Set=MFFS(Main_Data_Set,K,Steps,Alpha,Break_Value)
    return  Data_Set


# In[218]:


def result(Main_Labels,Dimension_Select_List):
    ACC_List=[]
    NMI_List=[]
    for i in Dimension_Select_List:
        Data_Set=Data_Sets(i)
        K_Means =K_Means_Clustering(Data_Set,Cluster_Count(Main_Labels))
        K_Labels=K_Means
        ACC_List.append(Acc(Main_Labels,K_Labels))
        NMI_List.append(NMI(Main_Labels,K_Labels))
       

    #----------- 
    #Show ACC in 2*D
    #-----------
    X_List=Dimension_Select_List
    Y_List=ACC_List
    plt.plot(X_List,Y_List,color='lightcoral',marker='D',markeredgecolor='black')
    plt.ylim(0,1) 
    plt.xlim(0,10) 
    plt.xlabel('Count Of Dimension') 
    plt.ylabel('Acc Value') 
    plt.title('ACC') 
    plt.show()
    #-----------  
    #End
    #-----------  
    
    
    
    #----------- 
    #Show Show NMI in 2*D
    #-----------
    X_List=Dimension_Select_List
    Y_List=NMI_List
    plt.plot(X_List,Y_List,color='lightcoral',marker='D',markeredgecolor='black')
    plt.ylim(0,1) 
    plt.xlim(0,10) 
    plt.xlabel('Count Of Dimension') 
    plt.ylabel('NMI Value') 
    plt.title('NMI') 
    plt.show()
    #-----------  
    #End
    #-----------


# In[216]:


def Data_Sets(K):
    return Multi_MFFS(K)
    #return Main_Data_Set


# In[223]:


Main_Data_Set = [
     [5,3,0,1],
     [4,0,0,1],
     [1,1,0,5],
     [1,0,0,4],
     [0,1,5,4],
    ]
Steps=10
Alpha=0.01
Break_Value=4


Main_Labels =[1,0,0,2,1]


Dimension_Select_List=[1,2,3,4]

result(Main_Labels,Dimension_Select_List)


# In[221]:


Main_Data_Set =pd.read_csv('C:\\Users\\babak_Nouri\\Desktop\\Main_Data_Set.CSV', header=None, skiprows=1)

Labels = pd.read_csv('C:\\Users\\babak_Nouri\\Desktop\\Main_Labels.CSV', header=None, skiprows=1)
Main_Labels=np.array(Labels).ravel().tolist()

Dimension_Select_List=[1,2,3,4]
   
    
result(Main_Labels,Dimension_Select_List)


# In[222]:


Main_Data_Set =pd.read_csv('C:\\Users\\babak_Nouri\\Desktop\\Cluster_Data.CSV', header=None, skiprows=1)

Labels = pd.read_csv('C:\\Users\\babak_Nouri\\Desktop\\Cluster_Label.CSV', header=None, skiprows=1)
Main_Labels=np.array(Labels).ravel().tolist()

Dimension_Select_List=[1,2,3,4]
   
    
result(Main_Labels,Dimension_Select_List)

