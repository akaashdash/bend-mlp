object Neuron { weights, bias }
object MLP { layer1, layer2, output }
object Vec { x, y }

def sigmoid(x):
  return 1.0 / (1.0 + 2.718281 ** (0-x))

def mlp_forward(layer1_weights, layer1_bias, layer2_weights, layer2_bias, output_weights, output_bias, input):
  open Vec: layer1_weights
  open Vec: layer2_weights
  open Vec: output_weights
  open Vec: input

  # Layer 1
  h1 = sigmoid(layer1_weights.x * input.x + layer1_weights.y * input.y + layer1_bias)

  # Layer 2
  h2 = sigmoid(layer2_weights.x * input.x + layer2_weights.y * input.y + layer2_bias)

  # Output layer
  return sigmoid(output_weights.x * h1 + output_weights.y * h2 + output_bias)

def mlp_backward(layer1_weights, layer1_bias, layer2_weights, layer2_bias, output_weights, output_bias, input, target, learning_rate):
  open Vec: layer1_weights
  open Vec: layer2_weights
  open Vec: output_weights
  open Vec: input

  # Forward pass to calculate hidden states
  h1 = sigmoid(layer1_weights.x * input.x + layer1_weights.y * input.y + layer1_bias)
  h2 = sigmoid(layer2_weights.x * input.x + layer2_weights.y * input.y + layer2_bias)
  output = sigmoid(output_weights.x * h1 + output_weights.y * h2 + output_bias)

  # Backpropagation
  d_output = output * (1.0 - output) * (target - output)

  # Output layer weight updates
  output_weights.x -= learning_rate * d_output * h1
  output_weights.y -= learning_rate * d_output * h2
  output_bias -= learning_rate * d_output

  # Hidden layer gradients
  d_h1 = h1 * (1.0 - h1) * (d_output * output_weights.x)
  d_h2 = h2 * (1.0 - h2) * (d_output * output_weights.y)

  # Hidden layer weight updates
  layer1_weights.x -= learning_rate * d_h1 * input.x
  layer1_weights.y -= learning_rate * d_h1 * input.y
  layer1_bias -= learning_rate * d_h1

  layer2_weights.x -= learning_rate * d_h2 * input.x
  layer2_weights.y -= learning_rate * d_h2 * input.y
  layer2_bias -= learning_rate * d_h2

  # Return updated weights and biases
  return (layer1_weights, layer1_bias, layer2_weights, layer2_bias, output_weights, output_bias)

def train_epochs(mlp, input, target, learning_rate, epochs):
  if epochs == 0:
    return mlp
  else:
    open MLP: mlp
    open Neuron: mlp.layer1
    open Neuron: mlp.layer2
    open Neuron: mlp.output
    (layer1_weights, layer1_bias, layer2_weights, layer2_bias, output_weights, output_bias) = mlp_backward(mlp.layer1.weights, mlp.layer1.bias, mlp.layer2.weights, mlp.layer2.bias, mlp.output.weights, mlp.output.bias, input, target, learning_rate)

    updated_mlp = MLP { 
        layer1: Neuron { weights: layer1_weights, bias: layer1_bias }, 
        layer2: Neuron { weights: layer2_weights, bias: layer2_bias }, 
        output: Neuron { weights: output_weights, bias: output_bias } 
    }

    # Recursively call train_epochs for the remaining epochs
    return train_epochs(updated_mlp, input, target, learning_rate, epochs - 1)

def main:
  mlp = MLP {
    layer1: Neuron { weights: Vec { x: 0.15, y: 0.2 }, bias: 0.35 },
    layer2: Neuron { weights: Vec { x: 0.25, y: 0.3 }, bias: 0.35 },
    output: Neuron { weights: Vec { x: 0.4, y: 0.45 }, bias: 0.6 }
  }

  input = Vec { x: 0.05, y: 0.1 }
  target = 0.01

  trained_mlp = train_epochs(mlp, input, target, 0.5, 3) # Train for 3 epochs
  return trained_mlp 