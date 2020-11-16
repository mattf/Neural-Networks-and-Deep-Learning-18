#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import matplotlib.pyplot as plt

def computeCost(X, y, theta):
    m = np.size(X[:,0])
    J = 1/(2*m)*np.sum((np.dot(X,theta)-y)**2)
    return J
data = np.loadtxt('C:/Users/李海峰/Desktop/人工神经网络/ex1data2.txt',delimiter=",",dtype="float")
m = np.size(data[:,0])
X = data[:,0:1]
y = data[:,1:2]
plt.plot(X,y,"rx")
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show()
one = np.ones(m)
X = np.insert(X,0,values=one,axis=1)
theta = np.zeros((2,1))
iterations = 1500
alpha = 0.01
J_history = np.zeros((iterations,1))
for iter in range(0,iterations):
    theta = theta - alpha/m*np.dot(X.T,(np.dot(X,theta)-y))
    J_history[iter] = computeCost(X,y,theta)
    
plt.plot(data[:,0],np.dot(X,theta),'-')
plt.show()

print(theta)
print(J_history)


# In[3]:


import numpy as np
import matplotlib.pyplot as plt


def computeCost(X, y, theta):
    m = np.size(X[:,0])   
    J = 1/(2*m)*np.sum((np.dot(X,theta)-y)**2)
    return J

 
data = np.loadtxt('C:/Users/李海峰/Desktop/人工神经网络/ex1data2.txt',delimiter=",",dtype="float")
m = np.size(data[:,0])

X = data[:,0:2]
y = data[:,2:3]

mu = np.mean(X,0)  
sigma = np.std(X,0)
X_norm = np.divide(np.subtract(X,mu),sigma)
one = np.ones(m)
X_norm = np.insert(X_norm,0,values=one,axis=1)

alpha = 0.05
iterations = 100
theta = np.zeros((3,1)); 
J_history = np.zeros((iterations,1))  
for i in range(0,iterations):
    theta = theta - alpha/m*np.dot(X_norm.T,(np.dot(X_norm,theta)-y))
    J_history[i] = computeCost(X_norm,y,theta)
print(theta)
x_col = np.arange(0,iterations)
plt.plot(x_col,J_history,'-b')
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()
print(J_history)

test1 = [1,1650,3]
test1[1:3] = np.divide(np.subtract(test1[1:3],mu),sigma)
price = np.dot(test1,theta)
print(price)     


# In[ ]:




