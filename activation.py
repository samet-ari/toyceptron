from layer import Layer

class Network:
    def __init__(self, input_size, activation):
        self.input_size = input_size #cest  la taille de input, dans notre cas 3
        self.activation = activation #on a pris la fonction d'activation en dehors
        self.layers = []             #on garde une liste de couches

    def add(self, weights, biases):   #il ajoute une couche à la liste de couches
        layer = Layer(weights, biases)
        self.layers.append(layer)   # il ajoute la liste de layers

    def feedforward(self, inputs):
        output = inputs
        for layer in self.layers:
            raw = layer.forward(output) #z=Wx+bias
            output = [self.activation(x) for x in raw] #On applique la fonction d'activation à chaque output
        return output                                  # for layer in self.layers: on avait deja defini dans la classe layer
