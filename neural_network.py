import random
import math
import sys


TARGET = 0.00
CALCULATED_RESULT = 0
OUTPUT_SUM = 0
HIDDEN_SUM = list()
NUMBER_OF_ITERATIONS = 0

class Neuron:
    def __init__(self , id = None , input = None):
        self.value = input
        self.id = id
        self.links_to = list()
        self.links_from = list()

    def __str__(self):
        retVal = "Neuron Id: " + str(self.id) + " Value: " + str(self.value)
        retVal += "\nLinks from: \n"
        for link in self.links_from:
            retVal += link.__str__() + "\n"

        retVal += "\nLinks to: \n"
        for link in self.links_to:
            retVal += link.__str__() + "\n"

        return retVal

    def __lt__(self , other):
        return self.id < other.id

    def addId(self , id : int):
        self.id = id

    def addValue(self , value: int):
        self.value = value

    def addValueAndId(self , value : int , id : int):
        self.value = value
        self.id = id

    def createNeuralLink(self , neuron , direction : bool , weight = None):
        if direction is True:
            self.links_from.append(NeuralLink(self , neuron , weight))
        else:
            self.links_to.append(NeuralLink(neuron , self , weight))

class NeuralLink:
    def __init__(self , source : Neuron , destination : Neuron , weight : int):
        self.source = source
        self.destination = destination
        self.weight = weight

    def __str__(self):
        return "Source Neuron: " + str(self.source.id) + " Destination Neuron: " + str(self.destination.id) + " Weight: " + str(self.weight)


class NeuralNetwork:
    def __init__(self , input = None , output = None , hidden = None):
        self.input = input
        self.output = output
        self.hidden = hidden           #All three are list that take arguments Neuron

        self.input.sort()
        self.output.sort()
        self.hidden.sort()

    def __str__(self):
        retVal = "Input Layer:\n"
        for input_neuron in self.input:
            retVal += input_neuron.__str__() + "\n"

        retVal += "\nHidden Layer:\n"
        for hidden_neuron in self.hidden:
            retVal += hidden_neuron.__str__() + "\n"

        retVal += "\nOutput Layer:\n"
        for output_neuron in self.output:
            retVal += output_neuron.__str__() + "\n"

        return retVal

    def generateLinks(self):
        for input_neuron in self.input:
            for hidden_neuron in self.hidden:
                input_neuron.createNeuralLink(hidden_neuron , True , random.random())

        for output_neuron in self.output:
            for hidden_neuron in self.hidden:
                output_neuron.createNeuralLink(hidden_neuron , False , random.random())

        for hidden_neuron in self.hidden:
            for input_neuron in self.input:
                hidden_neuron.createNeuralLink(input_neuron , False , random.random())
            for output_neuron in self.output:
                hidden_neuron.createNeuralLink(output_neuron , True , random.random())

    def activationFunction(self , x : float):
        return 1 / (1 + math.exp(-x))           #Sigmoid function

    def derivativeActivationFunction(self , x : float):
        return self.activationFunction(x) * (1 - self.activationFunction(x))        #Only for sigmoid activation function

    def updateLinks(self , neuron : Neuron):
        if neuron.links_from is not None:
            for link in neuron.links_from:
                link.source = neuron

        if neuron.links_to is not None:
            for link in neuron.links_to:
                link.destination = neuron

    def ForwardPropagation(self):
        global OUTPUT_SUM , HIDDEN_SUM
        #Initial hidden neuron values (hidden sum)
        for hidden_neuron in self.hidden:
            value = 0
            for link in hidden_neuron.links_to:
                value += link.source.value * link.weight
            hidden_neuron.value = value

        #Modified hidden neuron values
        for hidden_neuron in self.hidden:
            HIDDEN_SUM.append(hidden_neuron.value)
            hidden_neuron.value = self.activationFunction(hidden_neuron.value)

        #Initial result value (output sum)
        initial_result = 0
        for output_neuron in self.output:
            for link in output_neuron.links_to:
                initial_result += link.weight + link.source.value
        OUTPUT_SUM = initial_result

        #Final output
        final_result = self.activationFunction(initial_result)
        return final_result

    def BackPropagation(self):
        global TARGET , CALCULATED_RESULT

        output_margin_of_error = TARGET - CALCULATED_RESULT
        delta_output_sum = self.derivativeActivationFunction(OUTPUT_SUM) * output_margin_of_error

        #Ajusting weights
        #output - hidden
        hidden_to_outer_weights = list()
        for hidden_neuron in self.hidden:
            delta_weights = delta_output_sum * hidden_neuron.value
            new_weight = 0
            for link in hidden_neuron.links_from:
                hidden_to_outer_weights.append(link.weight)
                new_weight = delta_weights + link.weight
                link.weight = new_weight

            #Changing date in output list
            for output_neuron in self.output:
                for link in output_neuron.links_to:
                    if link.source is hidden_neuron:
                        link.weight = new_weight

        #hidden - input
        delta_hidden_sum = list()
        i = 0
        for hidden_neuron in self.hidden:
            for link in hidden_neuron.links_from:
                delta_hidden_sum.append(delta_output_sum * link.weight * self.derivativeActivationFunction(HIDDEN_SUM[i]))

        delta_weights_input = list()
        for input_neuron in self.input:
            for item in delta_hidden_sum:
                delta_weights_input.append(item * input_neuron.value)

        #Adding values
        for input_neuron in self.input:
            for link in input_neuron.links_from:
                link.weight = delta_weights_input.pop()

    def TrainNetwork(self):
        global NUMBER_OF_ITERATIONS , CALCULATED_RESULT , TARGET
        CALCULATED_RESULT = self.ForwardPropagation()
        while(abs(TARGET - CALCULATED_RESULT) > 0.001):
            self.BackPropagation()
            CALCULATED_RESULT = self.ForwardPropagation()
            NUMBER_OF_ITERATIONS += 1
            print("Iteration: " , NUMBER_OF_ITERATIONS)
            print("Current margin of error: " , (TARGET - CALCULATED_RESULT) * 100 , "%")
            print("Curretn result " , CALCULATED_RESULT)



#Input layer
input1 = Neuron(1 , 1)
input2 = Neuron(2 , 1)

input_layer = list()
input_layer.append(input1)
input_layer.append(input2)

#Output layer
output = Neuron(1 , 0)

output_layer = list()
output_layer.append(output)

#Hidden layer
hidden1 = Neuron(1)
hidden2 = Neuron(2)
hidden3 = Neuron(3)

hidden_layer = list()
hidden_layer.append(hidden1)
hidden_layer.append(hidden2)
hidden_layer.append(hidden3)

braniac_network = NeuralNetwork(input_layer , output_layer , hidden_layer)
braniac_network.generateLinks()

print("Input layer")
for neuron in input_layer:
    print(neuron.__str__())
print("\n")

print("Hidden layer")
for neuron in hidden_layer:
    print(neuron.__str__())
print("\n")

print("Output layer")
for neuron in output_layer:
    print(neuron.__str__())
print("\n")

print("Neural network: braniac")
print(braniac_network.__str__())

CALCULATED_RESULT = braniac_network.ForwardPropagation()
print("Neural network: braniac , After ForwardPropagation")
print(braniac_network.__str__())
print("CALCULATED RESULT: " , CALCULATED_RESULT)
print("OUTPUT SUM:" , OUTPUT_SUM)

braniac_network.BackPropagation()
print("Neural network: braniac , After BackPropagation")
print(braniac_network.__str__())

print("Training network...")
braniac_network.TrainNetwork()

print("Total Number of iterations:" , NUMBER_OF_ITERATIONS)
print("Nertwork after training: \n" , braniac_network.__str__())
