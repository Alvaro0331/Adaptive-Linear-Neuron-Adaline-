import numpy as np
import matplotlib.pyplot as plt

## Adaline
def adaline(X,d,activacion):
    #Inicializacion de pesos
    n_features=X.shape[1]
    bias=np.random.uniform(0,1)
    w=np.random.uniform(0,1,n_features)

    #Hyperparametros
    alpha=0.01
    epoch=1
    history=[]

    #Entrenamiento
    while True:
        global_error=False
        for i in range(len(X)):
            #Calculo de la salida
            z=np.dot(w,X[i])+bias
            y_pred=funcion_activacion(activacion,z)
            #Calculo del error
            error=d[i]-y_pred
            #Actualizacion de pesos y bias
            w+=alpha*error*X[i]
            bias+=alpha*error
            global_error=True
            if error!=0:
                global_error=True
        history.append((epoch,w.copy(),bias))
        if not global_error:
            break
        
        epoch+=1
    
    return history


##Funcion de activacion
def funcion_activacion(activacion,z):
    if activacion == "sigmoid":
        return 1/(1+np.exp(-z))
    elif activacion == "tanh":
        return np.tanh(z)
    else:
        return int(z>=0)