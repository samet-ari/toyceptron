class Neuron:
    #parameters
    def __init__(self, weights, bias): # cette methode il garde les poids et le biais du neurone
        self.weights = weights
        self.bias = bias
    def forward(self, inputs): # Cette methode, il calcule forward pass
        total = 0
        for w, x in zip(self.weights, inputs):
            total += w * x
        total += self.bias  #total=total+(wi​∗xi​)
                            #on dit "dot product"
                            #i=Sigma(wi​∗xi​)+b
        return total       # on retourne le total, c'est la valeur brute du neurone, avant activation

    