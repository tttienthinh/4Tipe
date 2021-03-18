import numpy as np
import matplotlib.pyplot as plt
"""
17/03/2021 tranthuongtienthinh
Feed Forward model pour résoudre le xor
algorithme procédural
"""

sigmoid = lambda x: 1/(1 + np.exp(-x))
d_sigmoid = lambda x: x * (1-x)

def affichage2d(datavertx, dataverty, datarougex, datarougey, epoch=0, finish=False):
    """
    Cette fonction a été crée par LUCAS RIGOT, modifié par 4T
    """
    plt.scatter(datavertx, dataverty, color="green", label="Vert = 1")
    plt.scatter(datarougex, datarougey, color="red", label="Rouge = 0")
    plt.plot([0, 1], [0.5, 0.5], color="blue")
    plt.plot([0.5, 0.5], [0, 1], color="blue")
    plt.title(f"We are at epoch : {epoch}")
    plt.legend()
    # version arret
    if finish:
        plt.show()
    else:
        # version pause 1 seconde
        plt.pause(.00_000_001)
        plt.clf() # efface l'ancien graphique

test_data = np.random.random((100, 2))


def forward(w1, w2, data):
    y1 = np.dot(data, w1)  # (4, 2) 4 entrées vont vers 2 neuronnes
    b = sigmoid(y1)  # (4, 2) fonction d'activation

    y2 = np.dot(b, w2)  # (4, ) 4 entrées vont vers le neuronne de sortie
    c = sigmoid(y2)  # (4, ) fonction d'activation
    return c


# training data
train_input = np.array([
    [0, 0], [1, 1],
    [0, 1], [1, 0]
])
train_output = np.array([0, 0, 1, 1])

adding_train_input = np.array([])
adding_train_output = []
if True:
    adding_train_input = np.random.random((10, 2))
    for i in range(10):
        data = adding_train_input[i]
        if (data[0] - .5) * (data[1] - .5) < 0:
            adding_train_output.append(1)
        else:
            adding_train_output.append(0)
adding_train_output = np.array(adding_train_output)


# creating random weigths
np.random.seed(2)
input_n = 2
hidden_n = 10
w1 = np.random.rand(input_n, hidden_n)
w2 = np.random.rand(hidden_n, )

print("Before trainning this is the Result")
c = forward(w1, w2, train_input)
for i in range(4):
    print(f"for the entry {train_input[i]} : {c[i]}")

print("------------------ Training to be better ------------------")
lr = 0.1
losses = []
affichage = True
x = adding_train_input
y = adding_train_output
for i in range(100):
    x = np.concatenate((train_input, x), axis=0)
    y = np.concatenate((train_output, y), axis=0)
for epoch in range(10_001):
    y1 = np.dot(x, w1) # (4, 2) 4 entrées vont vers 2 neuronnes
    b = sigmoid(y1) # (4, 2) fonction d'activation
    
    y2 = np.dot(b, w2) # (4, ) 4 entrées vont vers le neuronne de sortie
    c = sigmoid(y2) # (4, ) fonction d'activation
    loss = -(1/4)*np.sum(y*np.log(c)+(1-y)*np.log(1-c))
    losses.append(loss)
    
    # (4, ) Calcul de l'écart correspondant à (dE/dc)*(dc/dy2)
    e2 = 2*(c - y) * d_sigmoid(c)
    dw2 = np.dot(e2, b) # (2, ) changement de w2
    
    # (4, 2) Calcul de l'écart correspondant à ((dE/dc)*(dc/dy2)) * (dy2/db)*(db/dy1)
    e1 = np.dot(e2.reshape(len(e2), 1), w2.reshape(1, hidden_n)) * d_sigmoid(b)
    dw1 = np.dot(e1.T, x) # (2, 2) changement de w1
    
    w1 -= dw1.T * lr
    w2 -= dw2 * lr

    if affichage and epoch % 100 == 0:
        c = forward(w1, w2, test_data)
        datavertx, dataverty = (test_data[c>.5]).T
        datarougex, datarougey = (test_data[c<.5]).T
        affichage2d(datavertx, dataverty, datarougex, datarougey,
                    epoch=epoch, finish=epoch==100_000)



plt.plot(losses)
plt.xlabel("EPOCHS")
plt.ylabel("Loss value")

c = forward(w1, w2, train_input)
for i in range(4):
    print(f"for the entry {train_input[i]} : {c[i]}")

plt.show()

