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
            global_error=True
        
        history.append((epoch,w.copy(),bias))
        
        epoch+=1
    
    return history