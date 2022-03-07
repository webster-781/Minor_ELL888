
import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler,StandardScaler,OneHotEncoder,LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.decomposition import KernelPCA

def minmax(X : np.array):   # Normalises Feature matrix
    scaler = MinMaxScaler()
    scaler.fit(X)
    return scaler.transform(X)

def standard_scaler(X: np.array):    # Normalises Feature Matrix
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler.transform(X)

def getSpecialDistance(X: np.array):    # Special Distance
    n,d = X.shape
    D = np.zeros((n,n))
    missing = [[] for _ in range(n)]
    for i in range(n):
        missing[i] = np.where(X[i] == np.nan)

    for i in range(n):
        for j in range(i):
            miss = missing[i] + missing[j]
            x = np.delete(X[i],miss)
            y = np.delete(X[j],miss)
            D[i][j] = D[j][i] = np.linalg.norm(x-y)
    
    return D

# E - step algorithms
def KnnE(D : np.array, k : int = 2):
    # First we find the closest k neighbours of each node
    k = (int)(k)
    n = D.shape[0]
    m = np.max(D)
    for i in range(n):
        D[i][i] = m+1       # just to not get itself as a neighbour
    row = []
    col = []
    for i in range(n):
        neighbours = np.argpartition(D[i],k)[:k]
        row = row + list(np.ones(k)*i)
        col = col + list(neighbours)

    data = np.ones(k*n)

    for i in range(n):
        D[i][i] = 0        # just to reset to correct disimilarity values
    
    row = np.array(row, dtype = int)
    col = np.array(col, dtype = int)
    data = np.array(data)

    # print(row.shape,col.shape,data.shape)

    mat = csr_matrix((data, (row,col)),  shape=(n,n))
    mat = mat + mat.T
    mat[mat.nonzero()] = 1
    return mat


# W - step algorithms

def GaussianW(D: np.array, E : csr_matrix,sigma = 1):     # Given Dissimilarity matrix(Normalised in [0,1]) and Edge Matrix, returns the weight matrix for the corresponding graph using Gaussian assumption 
    n = D.shape[0]
    s = sigma*sigma

    for i in range(n):
        for j in E[i].indices:
            E[i,j] = np.exp(-D[i][j]*D[i][j]/(s))
    
    return E

# getting Dissimilarity matrix

def getD(X : np.array):     # All of type reals
    n,d = X.shape
    X = minmax(X)
    D = np.zeros((n,n))
    for i in range(n):
        for j in range(i):
            D[i][j] = D[j][i] = np.linalg.norm((X[i]-X[j]))
    return D


def getEfficiency(W : csr_matrix, y : np.array):
    n = W.get_shape()[0]
    acc = np.zeros((n))
    for i in range(n):
        ind = W[i].indices
        tot = np.sum(W[i])
        cnt = 0
        for j in ind:
            cnt += W[i,j]*(y[j] == y[i])
        acc[i] = cnt/tot
    return np.mean(acc)

def betterEfficiency(W : csr_matrix, y : np.array):
    n = W.get_shape()[0]
    acc = np.zeros((n))
    for i in range(n):
        ind = W[i].indices
        cnt = 0
        for j in ind:
            cnt += W[i,j]*(y[j] == y[i])
        tot = W[i].sum()
        acc[i] = cnt/tot
        if(acc[i] > 1/2):
            acc[i] = 1
        else:
            acc[i] = 0
    return np.mean(acc)

def HomeophilicEff(A : np.array, y : np.array):
    n = A.shape[0]
    c1 = 0
    c2 = 0
    for i in range(n):
        for j in A[i].indices:
            if(y[i] == y[j]):
                c1 += 1
            else:
                c2 +=1
    return c1/(c1+c2)

def getMissData(X : np.array , freq = 0.05):
    X = np.array(X,dtype = float)
    n,d = X.shape
    m = (int)(freq * n)

    for j in range(d):
        rng = np.random.default_rng()
        ind = rng.choice(n, size=m, replace=False)
        for i in ind:
            # print(i,j)
            X[i][j] = (float)(np.nan)
            # print(i,j)

    return X

def getMissLabels(y : np.array , freq = 0.05):
    y = np.array(y,dtype = float)
    n = y.shape[0]
    m = (int)(freq * n)
    rng = np.random.default_rng()
    indAbs = np.array(rng.choice(n, size=m, replace=False))
    np.sort(indAbs,axis = None)
    y1 = np.ones(n)
    y1[indAbs] = 0
    indPre = []
    for i in range(n):
        if(y1[i] == 1):
            indPre.append(i)
    indPre = np.array(indPre)
    return indPre,indAbs

def imputeMean(X : np.array): # Imputation using mean of other rowss
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(X)
    return imp.transform(X)

def imputeKNN(X : np.array, k = 5): # Imputation using the nearest eulclidean neighbours of given value
    imp = imputer = KNNImputer(n_neighbors=k, weights="uniform")
    return imputer.fit_transform(X)

def imputeLR(X : np.array):
    lr = LinearRegression()
    imp = IterativeImputer(estimator=lr, verbose=2, max_iter=30, tol=1e-10, imputation_order='roman')
    imp.fit(X)
    imp.transform(X)

def plot_values(D : np.array, y: np.array, title = "For Sample EEG data", K = range(1,11)):
    eff = [[],[]]
    for k in K:
        A = KnnE(D, k = k)
        # W = GaussianW(A, D)
        eff[0].append(HomeophilicEff(A,y))
        eff[1].append(betterEfficiency(A,y))

    plt.plot(K,eff[0],color = 'green')
    plt.plot(K,eff[1], color = 'blue')

    plt.xlabel("Number of Neighbours(k)")
    plt.ylabel("Efficiency of Graph")
    plt.title(title)
    plt.legend(['Homeophilic Efficiency','Classification Efficiency'])
    plt.show()

def getRealDist(X : np.array):
    X = minmax(X)
    results =  cdist(X,X,'euclidean')
    results = results/np.max(results)
    return results

def getRealDist2(X : np.array):
    X = minmax(X)
    results =  cdist(X,X,'cosine')
    results = results/np.max(results)
    return results

def getIntDist(X : np.array):
    X = minmax(X)
    results =  cdist(X,X,'euclidean')
    results = results/np.max(results)
    return results


def getIntDist2(X : np.array):
    X = minmax(X)
    results =  cdist(X,X,'cosine')
    results = results/np.max(results)
    return results

def getCatDist(X : np.array):
    n,d = X.shape
    D = np.zeros((n,n))
    for i in range(n):
        for j in range(i):
            D[i][j] = D[j][i] = np.sum([(X[i][k] != X[j][k]) for k in range(d)])/d
    D = D/np.max(D)
    return D

def getAllDist(R : np.array,I: np.array,C: np.array):
    D1 = getRealDist(R)
    D2 = getIntDist(I)
    D3 = getCatDist(C)
    d1 = R.shape[1]
    d2 = I.shape[1]
    d3 = C.shape[1]
    return (d1*D1 + d2*D2 + d3*D3)/(d1+d2+d3)


def getAllDist2(R : np.array,I: np.array,C: np.array):
    D1 = getRealDist(R)
    D2 = getIntDist(I)
    D3 = getCatDist(C)
    d1 = R.shape[1]
    d2 = I.shape[1]
    d3 = C.shape[1]
    return (d1*D1 + d2*D2 + 2*d3*D3)/(d1+d2+2*d3)

def all_effs(D : np.array, y: np.array, title = "For Sample EEG data", K = range(1,11)):
    eff = [[],[]]
    for k in K:
        A = KnnE(D, k = k)
        # W = GaussianW(A, D)
        eff[0].append(getEfficiency(A,y))
        eff[1].append(betterEfficiency(A,y))

    return eff

def visualise(X_transformed : np.array, y: np.array, A:csr_matrix):
    n = A.shape[0]
    cnt1 = 0
    cnt2 = 0
    for i in range(n):
        for j in A[i].indices:
            x_values = [X_transformed[i][0],X_transformed[j][0]]
            y_values = [X_transformed[i][1],X_transformed[j][1]]
            if(y[i] == y[j]):
                plt.plot(x_values,y_values,color = 'green')
                cnt1 += 1
            else:
                plt.plot(x_values,y_values,color = 'red')
                cnt2 += 1
    print(cnt1/(cnt1 + cnt2)*100,"%")
    plt.title("2D Visualization of the Graph")
    plt.show()

def Transform(X: np.array, kernel = 'linear', n = 2):
    transformer = KernelPCA(n_components=n, kernel=kernel)
    X_transformed = transformer.fit_transform(X)
    return X_transformed

def scatterPlot(X_transformed : np.array, y : np.array):
    classes = np.unique(y)
    str_classes = [f"Class {i}" for i in classes] 
    size = list(range(classes.shape[0]))
    ind = [[] for _ in classes]
    for i in size:
        print(type(i))
        ind[i] = np.argwhere(y==classes[i]).flatten()
    cols = ['red','blue','green','yellow','grey','purple','indigo','black']
    for i in size:
        plt.scatter(X_transformed[ind[i],0],X_transformed[ind[i],1],color = cols[i])
    plt.title("Visulaisation/Scatter Plot of all data points after PCA")
    plt.legend(str_classes)

def plot_Q2(A,indAb,XPr,yPr,X,y,k):
    A0 = A.copy()
    indAbs = indAb.copy()
    XPre = XPr.copy()
    yPre = yPr.copy()
    cnt1 = cnt2 = 0
    for i in indAbs:
        x0,y0 = X[i],y[i]
        d = np.linalg.norm(XPre-x0,axis = 1)
        neighbours = np.argpartition(d,k)[:k]
        cnt = 0.0
        for j in neighbours:
            cnt += y[j]/k
            A0[i,j] = 1
            A0[j,i] = 1
        XPre = np.concatenate((XPre,x0.reshape(1,x0.shape[0])),axis = 0)
        ypred = np.round(cnt)
        if(ypred == y[i]):
            cnt1 += 1
        else:
            cnt2 += 1
        yPre = np.concatenate((yPre,[y[i]]),axis = 0)
        print(yPre.shape)
    visualise(A0,yPre)

    return [HomeophilicEff(A0,yPre),betterEfficiency(A0,yPre),cnt1/(cnt1+cnt2)]