#%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from numpy import dot


df1 = pd.read_csv('ratings.csv')
df2 = pd.read_csv('movies.csv')

df = pd.merge(df1,df2,on=['movieId'])

A = np.array(df.pivot_table( columns=['movieId'], index=['userId'], values='rating').fillna(0)) #matrix array A

n_u = len(df["userId"].unique())
n_m = len(df["movieId"].unique())
sparsity = len(df)/(n_u*n_m)
print("sparsity of ratings is %.2f%%" %(sparsity*100))   # calculate sparsity of data


plt.figure(figsize=(14,12))
plt.spy(A)
plt.title("Sparse Matrix")            

n_factors = 20              # Laten Factors

n_users , n_items = A.shape         

print("n_users = ",n_users, "||", "n_movies = ", n_items )

P = np.random.normal(scale=1./n_factors, size=(n_users, n_factors))   # Laten factor P
Q = np.random.normal(scale=1./n_factors, size=(n_items, n_factors))    # Laten Factor Q
print(P.shape , Q.shape)


user_bias = np.zeros(n_users,dtype=object)            
item_bias = np.zeros(n_items,dtype=object)
rating_bias = np.mean(A[np.where(A != 0)])         # Rating bias
learn_rate = 0.001                                 # learning rate for sgd
const = 0.01                                         # lamda costant
users,items = A.nonzero()         
n_epochs = 60                                        # no of iterations

def get_mse(pred, actual):                               # check rmse  by A - prediction (P dot Q.T)
    mse = mean_squared_error(pred,actual)
    return np.sqrt(mse)

error = []
for epochs in range(n_epochs):
    for u, i in zip(users, items):
        e = A[u, i] - dot(P[u, :], Q[i, :].T)

        # user_bias[i] += learn_rate * (e - const * U_B[i] )
        # item_bias[j] += learn_rate * (e - const * I_B[j])

        P[u, :] += learn_rate * (dot(e, Q[i, :]) - const * P[u, :])      #Derative for P matrix
        Q[i, :] += learn_rate * (dot(e, P[u, :]) - const * Q[i, :])        # Derative for Q matrix

    error.append(get_mse(dot(P, Q.T), A))
    print(f' itearation error {epochs + 1 } and error is : {error[epochs]}')




print("Matrix A",A)

print("P x Q" , dot(P,Q.T))                  

print(error)
fig, ax = plt.subplots()
ax.plot(error, color="g", label='Training RMSE')
ax.legend()
plt.show()