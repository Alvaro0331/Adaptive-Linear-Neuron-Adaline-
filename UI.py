import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import numpy as np
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import TextBox
from adaline import adaline


# Creacion de la figura
def crear_figura():
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes([0.1, 0.1, 0.6, 0.8]) # [left, bottom, width, height]
    ax.set_title("Práctica 3: Adaline")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel("X2", fontsize=12)
    ax.set_ylabel("X1", fontsize=12)
    #Dibujo de ejes
    ax.axhline(y=0, color='black', linewidth=0.7)
    ax.axvline(x=0, color='black', linewidth=0.7)
    global stateText, errorText
    stateText = ax.text(0.02, 0.95, "", transform=ax.transAxes)
    errorText = ax.text(0.02, 0.90, "", transform=ax.transAxes)
    
    #Leyenda
    leyenda = [
        plt.Line2D([0], [0], marker='o', color='w', label='Clase A', markerfacecolor='blue', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='Clase B', markerfacecolor='red', markersize=8),
        plt.Line2D([0], [0], linestyle='-', color='green', label='Frontera'),
    ]
    ax.legend(handles=leyenda, loc='upper right', fontsize=8)
    return fig, ax

#Creacion de los widgets
def crear_widgets(fig):
    #Texto de clase
    class0Text=ax.text(0.75, 0.85, "Click izquierdo:", transform=fig.transFigure, fontsize=10, color='black')
    class0Text = ax.annotate(" Clase A",xycoords=(class0Text),xy=(1, 0), verticalalignment='bottom', fontsize=10, color='blue')
    class1Text=ax.text(0.75, 0.75, "Click derecho:", transform=fig.transFigure, fontsize=10, color='black')
    class1Text = ax.annotate(" Clase B",xycoords=(class1Text),xy=(1, 0), verticalalignment='bottom', fontsize=10, color='red')
    #Textbox
    stopBox=widgets.TextBox(plt.axes([0.79, 0.65, 0.1, 0.05]), 'Stop:', initial="0.01")
    #RaddioButton para elegir la funcion de activacion
    radioAx=plt.axes([0.75, 0.45, 0.1, 0.1])
    radioAx.set_title("Activación", fontsize=10)
    radioActivation=widgets.RadioButtons(radioAx,('Sigmoid', 'Tanh', 'Step'))
    # Botones
    plotButton=widgets.Button(plt.axes([0.75, 0.25, 0.1, 0.15]), 'Train', color='lightblue', hovercolor='skyblue')
    clearButton=widgets.Button(plt.axes([0.75, 0.15, 0.1, 0.05]), 'Clear', color='lightcoral', hovercolor='salmon')
    
    return plotButton, clearButton, stopBox, radioActivation

#Evento para agregar puntos con el mouse
puntos = []  # Lista para almacenar los puntos clickeados
markers = [] # Lista para almacenar los objetos de los puntos dibujados
lineas = [] # Lista para almacenar la línea dibujada
etiquetas = [] # Lista para almacenar las etiquetas de los puntos
def onclick(event):
    if event.inaxes == ax:
        if event.button is MouseButton.LEFT:
            x, y = event.xdata, event.ydata
            marker, = ax.plot(x, y, 'bo')  # Dibuja un punto azul en la posición clickeada
            puntos.append((x, y))  # Agrega el punto a la lista de puntos
            markers.append(marker)  # Agrega el objeto del punto a la lista de markers
            etiquetas.append(1)  # Agrega la etiqueta correspondiente a la lista de etiquetas
            fig.canvas.draw()  # Actualiza la figura para mostrar el nuevo punto
        elif event.button is MouseButton.RIGHT:
            x, y = event.xdata, event.ydata
            marker, = ax.plot(x, y, 'ro')  # Dibuja un punto rojo en la posición clickeada
            puntos.append((x, y))  # Agrega el punto a la lista de puntos
            markers.append(marker)  # Agrega el objeto del punto a la lista de markers
            etiquetas.append(0)  # Agrega la etiqueta correspondiente a la lista de etiquetas
            fig.canvas.draw()  # Actualiza la figura para mostrar el nuevo punto


#Evento para limpiar los puntos
def clear(event):
    stateText.set_text("")  # Limpia el texto de estado
    errorText.set_text("")  # Limpia el texto de error
    for marker in markers:
        marker.remove()  # Elimina el punto de la figura
    for etiqueta in etiquetas:
        etiquetas.remove(etiqueta)  # Elimina la etiqueta de la lista de etiquetas
    markers.clear()  # Reinicia la lista de markers
    etiquetas.clear()
    puntos.clear() # Reinicia la lista de puntos
    clear_line()  # Limpia la línea de decisión

#Funcion para limpiar la linea
def clear_line():
    for linea in lineas:
        linea.remove()  # Elimina la línea de la figura
    lineas.clear()  # Reinicia la lista de líneas
    fig.canvas.draw()  # Actualiza la figura para reflejar los cambios

#Funcion para ejecutar el entrenamiento
def train(event):
    stop=float(stopBox.text)  # Obtiene el valor de stop del TextBox
    activacion=radioActivation.value_selected.lower()  # Obtiene la función de activación seleccionada
    x=np.array(puntos, dtype=float)
    d=np.array(etiquetas, dtype=int)
    if len(x) < 2:
        return
    
    historial, w, bias = adaline(x, d, activacion, stop)
    for epoch, w_epoca, b_epoca, error_epoca in historial:
        clear_line()
        m,c,x_vertical=calcular(w_epoca[0], w_epoca[1], b_epoca)
        stateText.set_text(f'Epoch: {epoch}')
        errorText.set_text(f'Error: {sum(error_epoca):.4f}')

        if x_vertical is not None:
            linea = ax.axvline(x=x_vertical, color='g', linewidth=1.5)
            lineas.append(linea)
        elif m is not None:
            x_vals=np.linspace(-10, 10, 200)
            y_vals=m*x_vals+c
            linea, = ax.plot(x_vals, y_vals, 'g-', linewidth=1.5,)  # Dibuja la línea de decisión
            lineas.append(linea)
        fig.canvas.draw_idle()  # Actualiza la figura para mostrar los cambios
        plt.pause(0.15)  # Pausa para visualizar el proceso de entrenamiento

#Funcion para calcular m y c
def calcular(w0,w1,bias):
    if np.isclose(w1, 0.0):
        if np.isclose(w0, 0.0):
            return None, None, None
        x_vertical = -bias / w0
        return None, None, x_vertical
    m = -w0 / w1
    c = -bias / w1
    return m, c, None


fig, ax = crear_figura()
plotButton, clearButton, stopBox, radioActivation = crear_widgets(fig)
fig.canvas.mpl_connect('button_press_event', onclick)
clearButton.on_clicked(clear)
plotButton.on_clicked(train)
plt.show()