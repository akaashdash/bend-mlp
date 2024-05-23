# Simple MLP in Bend

[Bend](https://github.com/HigherOrderCO/Bend) is a recently released parallel programming language. I wanted to mess around with it and see what I could create, so I created a simple MLP.

## Usage

Currently Bend doesn't support IO operations, so loading custom datasets isn't a possibility right now. You can modify the weights, biases, inputs, and outputs to see what you can come up with.

Run using one of the Bend run commands. Since I am on a Mac, I use:
```
bend run-c mlp.bend  
```

## Results

Currently Bend doesn't support converting numbers to strings or many other IO operations, so the output isn't pretty. Here is what we get running the program as is:
```
Result: (train_epochs (MLP (Neuron (Vec 0.150 0.200) 0.350) (Neuron (Vec 0.250 0.300) 0.350) (Neuron (Vec 0.400 0.450) 0.600)) (Vec 0.050 0.100) 0.010 0.500 3)
```

Not sure why the list of epoch losses looks like that but I'm trying to fix it.