# Define objects to represent our data structures
object Neuron { weights, bias }  # A single neuron with weights and a bias
object MLP { layer1, layer2, output }  # The Multi-Layer Perceptron (MLP) with three neurons
object Vec { x, y } # A 2D vector for input and weights

# Use a 24-bit approximation of Euler's number 
# Bend only supports 24-bit numbers
E = 2.718281

# Sigmoid activation function
def sigmoid(x):
  return 1.0 / (1.0 + E ** (0-x))

# Forward pass through the MLP
def mlp_forward(layer1_weights, layer1_bias, layer2_weights, layer2_bias, output_weights, output_bias, input):
  # Open the Vec objects to access their fields directly
  open Vec: layer1_weights
  open Vec: layer2_weights
  open Vec: output_weights
  open Vec: input

  # Calculate the activation for the first hidden neuron
  h1 = sigmoid(layer1_weights.x * input.x + layer1_weights.y * input.y + layer1_bias)

  # Calculate the activation for the second hidden neuron
  h2 = sigmoid(layer2_weights.x * input.x + layer2_weights.y * input.y + layer2_bias)

  # Calculate the output neuron's activation (the prediction)
  return sigmoid(output_weights.x * h1 + output_weights.y * h2 + output_bias)

# Backward pass to update weights and biases
def mlp_backward(layer1_weights, layer1_bias, layer2_weights, layer2_bias, output_weights, output_bias, input, target, learning_rate):
  # Open the Vec objects to access their fields directly
  open Vec: layer1_weights
  open Vec: layer2_weights
  open Vec: output_weights
  open Vec: input

  # Perform a forward pass to get activations
  h1 = sigmoid(layer1_weights.x * input.x + layer1_weights.y * input.y + layer1_bias)
  h2 = sigmoid(layer2_weights.x * input.x + layer2_weights.y * input.y + layer2_bias)
  output = sigmoid(output_weights.x * h1 + output_weights.y * h2 + output_bias)

  # Calculate the Mean Squared Error loss
  loss = (target - output) ** 2.0 

  # Backpropagation

  # Calculate the gradient of the output neuron
  d_output = output * (1.0 - output) * (target - output)

  # Update the output neuron's weights and bias
  output_weights.x -= learning_rate * d_output * h1
  output_weights.y -= learning_rate * d_output * h2
  output_bias -= learning_rate * d_output

  # Calculate the gradients for the hidden neurons
  d_h1 = h1 * (1.0 - h1) * (d_output * output_weights.x)
  d_h2 = h2 * (1.0 - h2) * (d_output * output_weights.y)

  # Update the hidden neurons' weights and biases
  layer1_weights.x -= learning_rate * d_h1 * input.x
  layer1_weights.y -= learning_rate * d_h1 * input.y
  layer1_bias -= learning_rate * d_h1

  layer2_weights.x -= learning_rate * d_h2 * input.x
  layer2_weights.y -= learning_rate * d_h2 * input.y
  layer2_bias -= learning_rate * d_h2

  # Return updated weights, biases, and the loss
  return (layer1_weights, layer1_bias, layer2_weights, layer2_bias, output_weights, output_bias, loss)

# Function to train the MLP for a specified number of epochs
def train_epochs(mlp, input, target, learning_rate, epochs):
  if epochs == 0:
    # If all epochs are done, perform a forward pass to get the final prediction
    open MLP: mlp
    open Neuron: mlp.layer1
    open Neuron: mlp.layer2
    open Neuron: mlp.output
    prediction = mlp_forward(mlp.layer1.weights, mlp.layer1.bias, mlp.layer2.weights, mlp.layer2.bias, 
                              mlp.output.weights, mlp.output.bias, input)
    
    # Return the trained MLP, an empty loss list, and the prediction
    return (mlp, [], prediction) 
  else:
    # If more epochs to go, open the MLP and Neuron objects to access fields
    open MLP: mlp
    open Neuron: mlp.layer1
    open Neuron: mlp.layer2
    open Neuron: mlp.output
    # Perform one epoch of backpropagation and get updated weights, biases, and loss
    (layer1_weights, layer1_bias, layer2_weights, layer2_bias, output_weights, output_bias, loss) = mlp_backward(mlp.layer1.weights, mlp.layer1.bias, mlp.layer2.weights, mlp.layer2.bias, mlp.output.weights, mlp.output.bias, input, target, learning_rate)

    # Create a new MLP with the updated weights and biases
    updated_mlp = MLP { 
        layer1: Neuron { weights: layer1_weights, bias: layer1_bias }, 
        layer2: Neuron { weights: layer2_weights, bias: layer2_bias }, 
        output: Neuron { weights: output_weights, bias: output_bias } 
    }

    # Recursively train for the remaining epochs
    (trained_mlp, losses, prediction) = train_epochs(updated_mlp, input, target, learning_rate, epochs - 1)
    
    # Add the current epoch's loss to the beginning of the list
    return (trained_mlp, [loss] + losses, prediction) 

# The main function of our program
def main:
  # Initialize the MLP with some weights and biases
  mlp = MLP {
    layer1: Neuron { weights: Vec { x: 0.15, y: 0.2 }, bias: 0.35 },
    layer2: Neuron { weights: Vec { x: 0.25, y: 0.3 }, bias: 0.35 },
    output: Neuron { weights: Vec { x: 0.4, y: 0.45 }, bias: 0.6 }
  }

  # Set the input vector
  input = Vec { x: 0.05, y: 0.1 }
  # Set the target output
  target = 0.01

  # Train the MLP for 3 epochs with a learning rate of 0.5
  (trained_mlp, losses, prediction) = train_epochs(mlp, input, target, 0.5, 3) 

  # Return the trained MLP, the losses for each epoch, and the final prediction
  # I'm still trying to figure out how to get this to print better
  return (trained_mlp, losses, prediction)  