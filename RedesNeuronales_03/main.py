import random



from RedesNeuronales_03.Perceptron import Perceptron
from RedesNeuronales_03.SigmoidNeuron import SigmoidNeuron


def main ():
    #Creamos el Perceptron
    perceptron = Perceptron(0.01)

    #Funcion que se ocupa para obtener el output correcto (en este caso la recta)
    func = lambda x,y: 2*x+3<=y
    print(func(2,1))
    #Se entrena al perceptron 200 veces, con la funcion func
    perceptron.train(1000,func)
    #Se prueba el Perceptron con 100 iteraciones
    perceptron.run(1000)
    #Se grafican los resultados obtenidos en run()
    plt = perceptron.plotResults("Perceptron","Learning Perceptron")


    #Se crea una Sigmoid Neuron
    sigmoid_neuron = SigmoidNeuron(0.5)

    #Se entrena iterando 200 veces
    sigmoid_neuron.train(1000,func)
    #Se prueba la Sigmoid neuron con 100 casos
    sigmoid_neuron.run(1000)
    #Se grafican los resultados del entrenamiento
    plt2=sigmoid_neuron.plotResults("Sigmoid Neuron","Learning Sigmoid Neuron")

    #Se muestran los graficos de resultados del Perceptron y la Sigmoid Neuron
    plt.show()
    plt2.show()










main()

