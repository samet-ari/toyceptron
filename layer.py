from neuron import Neuron

class Layer:
    def __init__(self, weights_list, biases_list):
        self.neurons = [] #j'ai cree une liste de neurones vide

        for weights, bias in zip(weights_list, biases_list):
            self.neurons.append(Neuron(weights, bias))  #chaque itration, on ajoute un neurone à la liste de neurones, avec les poids et le biais

    def forward(self, inputs): #cette methode il envoie les inputs à chaque neurone et puis retourne une liste de leurs outputs
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.forward(inputs))  #0.2+(−0.2)+1.6+0.0=1.6
                                                    #(−0.4*1.0)+(0.3*2.0)+(0.1*4.0)+0.1 (bias) =0.7

                                                    # il compose que deux neurones mais on put ajouter autant de neurones que l'on veut
        return outputs  #[1.6, 0.7]   #c'est la valeur brute du neurone aussi avant activation
