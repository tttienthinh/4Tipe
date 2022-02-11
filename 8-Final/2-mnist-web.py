#Loading
from tensorflow.keras.datasets import mnist # Seulement pour importer les images
import matplotlib.pyplot as plt
import numpy as np

# https://towardsdatascience.com/softmax-regression-in-python-multi-class-classification-3cb560d90cb2


(train_X, train_y), (test_X, test_y) = mnist.load_data()#Plotting
fig = plt.figure(figsize=(10,7))
for i in range(15):  
    ax = fig.add_subplot(3, 5, i+1)
    ax.imshow(train_X[i], cmap=plt.get_cmap('gray'))
    ax.set_title('Label (y): {y}'.format(y=train_y[i]))
    plt.axis('off')
plt.show()

X_train = train_X.reshape(60000,28*28)
X_test = test_X.reshape(10000,28*28)

train_y

def one_hot(y, c):
    
    # y--> label/ground truth.
    # c--> Number of classes.
    
    # A zero matrix of size (m, c)
    y_hot = np.zeros((len(y), c))
    
    # Putting 1 for column where the label is,
    # Using multidimensional indexing.
    y_hot[np.arange(len(y)), y] = 1
    
    return y_hot

def softmax(z):
    
    # z--> linear part.
    
    # subtracting the max of z for numerical stability.
    exp = np.exp(z - np.max(z))
    
    # Calculating softmax for all examples.
    for i in range(len(z)):
        exp[i] /= np.sum(exp[i])
        
    return exp


def fit(X, y, lr, c, epochs):
    
    # X --> Input.
    # y --> true/target value.
    # lr --> Learning rate.
    # c --> Number of classes.
    # epochs --> Number of iterations.
    
        
    # m-> number of training examples
    # n-> number of features 
    m, n = X.shape
    
    # Initializing weights and bias randomly.
    w = np.random.random((n, c))
    b = np.random.random(c)    # Empty list to store losses.
    losses = []
    
    # Training loop.
    for epoch in range(epochs):
        
        # Calculating hypothesis/prediction.
        z = X@w + b
        y_hat = softmax(z)
        
        # One-hot encoding y.
        y_hot = one_hot(y, c)
        
        # Calculating the gradient of loss w.r.t w and b.
        w_grad = (1/m)*np.dot(X.T, (y_hat - y_hot)) 
        b_grad = (1/m)*np.sum(y_hat - y_hot)
        
        # Updating the parameters.
        w = w - lr*w_grad
        b = b - lr*b_grad
        
        # Calculating loss and appending it in the list.
        loss = -np.mean(np.log(y_hat[np.arange(len(y)), y]))
        losses.append(loss)        # Printing out the loss at every 100th iteration.
        if epoch%100==0:
            print('Epoch {epoch}==> Loss = {loss}'
                  .format(epoch=epoch, loss=loss))    
    return w, b, losses

X_train = train_X.reshape(60000,28*28)# Normalizing. 
X_train = X_train/255# Training
w, b, l = fit(X_train, train_y, lr=0.9, c=10, epochs=200)

def predict(X, w, b):
    
    # X --> Input.
    # w --> weights.
    # b --> bias.
    
    # Predicting
    z = X@w + b
    y_hat = softmax(z)
    
    # Returning the class with highest probability.
    return np.argmax(y_hat, axis=1)

def accuracy(y, y_hat):
    return np.sum(y==y_hat)/len(y)



# Accuracy for training set.
train_preds = predict(X_train, w, b)
print(f"{accuracy(train_y, train_preds)} acc train")

# Accuracy for test set.# Flattening and normalizing.
X_test = test_X.reshape(10000,28*28)
X_test = X_test/255
test_preds = predict(X_test, w, b)
accuracy(test_y, test_preds)

fig = plt.figure(figsize=(15,10))
for i in range(40):  
    ax = fig.add_subplot(5, 8, i+1)
    ax.imshow(test_X[i], cmap=plt.get_cmap('gray'))
    
    ax.set_title('y: {y}'.format(y=test_y[i], y_hat=test_preds))
    plt.axis('off')
plt.show()