import numpy as np

## Adaline
def adaline(X,d,activacion,stop):
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
    while Error[-1] > stop:
        error=[]
        for i in range(X.shape[0]):
            #Calculo de la salida
            z=np.dot(w,X[i])+bias
            y_pred=funcion_activacion(activacion,z)
            #Actualizacion de pesos y bias
            w+=alpha*(d[i]-y_pred)*X[i]
            bias+=alpha*(d[i]-y_pred)
            #Calculo del error
            error.append((d[i]-y_pred)**2)
        Error.append(sum(error))
        history.append((epoch,w.copy(),bias,error))
        epoch+=1
    
    return history, w, bias


##Funcion de activacion
def funcion_activacion(activacion,z):
    if activacion == "sigmoid":
        return 1/(1+np.exp(-z))
    elif activacion == "tanh":
        return np.tanh(z)
    else:
        return int(z>=0)
    
##Datos de prueba
#Input dataset
x = np.array([[-1,-1],
              [-1,1],
              [1,-1],
              [1,1]])
# Target values
d = np.array([-1, -1, -1, 1])

history, w, bias = adaline(x, d, "sigmoid", 0.001)
print('weight :',w)
print('Bias :',bias)