# Demo MNIST application

This is a feed forward neural network demo application for handwritten digit recognition on MNIST database.

The main goal was to implement the backpropagation algorithm in the simplest way.

## Usage ##

1. Install Java and Maven

2. Create a folder under `models` which contains a `model.properties` file with the settings. (Number of layer, number of neurons per layer, learning rate)

3. Build the application: `mvn clean install`

4. Train the model: `train <modelFolder>`

The app will train an epoch (60.000 examples in the MNIST database) and will ask you whether continue with the next epoch or stop the training.

If the training is stopped it will save the parameters into a subfolder, named as `parameters-sysdatetime` 

If you have an already trained model, you can train it further, or you can test the success rate of that.

Test only: `test modelFolder parametersFolder`

Train further: `train modelFolder parametersFolder`

## Source of the MNIST database ##

- https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/

I removed the header and split the training data into three files to avoid the GitHub upload limit.