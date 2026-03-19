import numpy as np

## Adaline
def adaline(X,d,stop, max_epochs=200):
    #Inicializacion de pesos
    n_features=X.shape[1]
    bias=np.random.uniform(0,1)
    w=np.random.uniform(0,1,n_features)
    
    
    #Hyperparametros
    alpha=0.01
    epoch=1
    history=[]
    Error=[stop+1]
    #Entrenamiento
    while Error[-1] > stop and epoch <= max_epochs:
        error=[]
        for i in range(X.shape[0]):
            #Calculo de la salida
            z=np.dot(w,X[i])+bias
            #Actualizacion de pesos y bias
            e=d[i]-z
            w+=alpha*e*X[i]
            bias+=alpha*e
            #Calculo del error
            error.append(e**2)
        Error.append(sum(error) / X.shape[0])
        history.append((epoch,w.copy(),bias,error))
        epoch+=1
    
    return history, w, bias


##Funcion de activacion
def funcion_activacion(activacion,z):
    if activacion == "sigmoid":
        return 1/(1+np.exp(-z))
    elif activacion == "tanh":
        return np.tanh(z)
    elif activacion == "relu":
        return np.maximum(0, z)
    elif activacion == "linear":
        return z
    else:
        return int(z>=0)

""" ##Datos de prueba
#Input dataset
x = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])
# Target values
clases = np.array([0, 0, 0, 1])
history, w, bias = adaline(x, clases, "tanh", 0.001)
print('weight :',w)
print('Bias :',bias) """