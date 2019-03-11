import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def values(X,w1,w2):
    Y = []
    for i in X:
        Y.append(w1*i+w2)
    return Y

def val(x,w1,w2):
    return w1*x+w2

def meanSquaredLoss(truth,guess):
    sum = 0
    for i in range(0,len(truth)):
        sum = sum + (truth[i]-guess[i])**2
    return sum/len(truth)

def squaredError(y1,y2):
    return (y1-y2)**2

def partialSlope(w1,w2,x,y):
    return (-2*x)*(y-(w1*x)-w2)

def partialIntercept(w1,w2,x,y):
    return (-2)*(y-(w1*x)-w2)

#global variables
X = [1,3,5,6,7,8,11,15,20,22,16,30]
Y = [2,6,2,9,10,28,26,33,33,44,35,65]
W1 = 0.01
W2 = 0.01
learning_rate = 0.00001
funcX = range(0,60)
errorStep = []
errorlist = []
fig = plt.figure("Liniar Regression - Stochastic Gradient Decent")
ax1 = fig.add_subplot(2,1,1)
ax1.set_title('Trained Model')
ax2 = fig.add_subplot(2,1,2)
ax2.set_ylabel('MSE')
ax2.set_xlabel('Epoch')
ax1.set_xlim([0, 50])
ax1.set_ylim([0, 50])
ax2.set_xlim([0, 125])
ax2.set_ylim([0, 800])
ax2.set_xticks(range(0, 125, 10))
funcY = values(funcX,W1,W2)
ax1.plot(X,Y,'ro')
ax1.plot(funcX,funcY)
loss = 999999999999


def animate(b):
    global W1,funcX,W2,errorlist,errorStep,X,Y,ani,loss
    for i in range(0,len(X)):
        x_curr = X[i]
        y_real = Y[i]
        w1Grad = partialSlope(W1,W2,x_curr,y_real)
        w2Grad = partialIntercept(W1,W2,x_curr,y_real)
        prevW1 = W1
        prevW2 = W2
        W1 = prevW1 - (learning_rate*w1Grad)
        W2 = prevW2 - (learning_rate*w2Grad) 
    ax1.clear()
    ax2.clear()
    ax1.set_title('Trained Model')
    ax2.set_ylabel('MSE')
    ax2.set_xlabel('Epoch')
    ax2.set_xticks(range(0, 125, 10))
    #print("Slope {} Intercept {}".format(W1,W2))
    ax1.set_xlim([0, 50])
    ax1.set_ylim([0, 50])
    ax2.set_xlim([0, 125])
    ax2.set_ylim([0, 800])
    guesses = values(X,W1,W2)
    error = meanSquaredLoss(Y,guesses)
    errorlist.append(error)
    errorStep.append(b)
    #print("Mean Sum Squared Error: {}".format(error))
    funcY = values(funcX,W1,W2)
    ax1.plot(X,Y,'ro')
    ax1.plot(funcX,funcY)
    ax2.plot(errorStep,errorlist)
    if(abs(loss - error) < 0.001 and abs(loss - error) < 0.001):
            ani.event_source.stop()
    else:
        loss = error

ani = animation.FuncAnimation(fig, animate, interval=100,repeat=False)
plt.show()

    

