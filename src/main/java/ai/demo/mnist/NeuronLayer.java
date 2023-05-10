package ai.demo.mnist;

import ai.demo.mnist.activation.Activation;

/**
 * Represents a single layer in a feed forward neural network
 */
public class NeuronLayer
{
	private final int inputCount;
	private final int neuronCount;

	private final float[][] weights;
	private final float[] biases;

	private final Activation activation;

	private float[] inputs;
	private float[] outputs;
	
	private float[] errors;

	/**
	 * Constructor
	 */
	public NeuronLayer(int inputCount, int neuronCount, float[][] weights, float[] biases, Activation activation)
	{
		this.inputCount = inputCount;
		this.neuronCount = neuronCount;
		this.weights = weights;
		this.biases = biases;
		this.activation = activation;
	}

	/**
 	 * Calculates the output of a neural layer
	 * Applies weights and biases and executes the activation function on all neurons
	 * @param inputs the inputs
	 * @return the outputs
	 */
	public float[] feedForward(float[] inputs)
	{
		// Save the inputs
		this.inputs = inputs;

		// Collector of the outputs
		float[] outputs = new float[neuronCount];

		// Iterating over on all neurons
		for (int neuron = 0; neuron < neuronCount; neuron++)
		{
			// Iterating over on all inputs
			for (int input = 0; input < inputCount; input++)
			{
				// Apply weight
				outputs[neuron] += weights[neuron][input] * inputs[input];
			}

			// Apply bias
			outputs[neuron] += biases[neuron];

			// Apply activation function
			outputs[neuron] = activation.forward(outputs[neuron]);
		}

		// Save the outputs
		this.outputs = outputs;

		return outputs;
	}

	/**
	 * Back-propagates the errors from the output to the output of the previous layer
	 * @param outputErrors the errors at the output
	 * @return the errors at the output of the previous layer
	 */
	public float[] backPropagateErrors(float[] outputErrors)
	{
		// Back-propagate the errors from the output to the point before the activation function
		// We have to store these errors, because it will be used when the weights will be updated
		errors = new float[neuronCount];
		for (int neuron = 0; neuron < neuronCount; neuron++)
		{
			errors[neuron] = activation.gradient(outputs[neuron]) * outputErrors[neuron];
		}

		// Back-propagate the errors to the output of the previous layer
		// This will be the return value, so the same back-propagate process can be repeated at the previous layer
		float[] prevOutputErrors = new float[inputCount];
		for (int input = 0; input < inputCount; input++)
		{
			for (int neuron = 0; neuron < neuronCount; neuron++)
			{
				prevOutputErrors[input] += weights[neuron][input] * errors[neuron];
			}
		}

		return prevOutputErrors;
	}

	/**
	 * Updates the parameters (weights and biases) using the already back-propagated errors
	 * @param learningRate the learning rate
	 */
	public void updateParameters(double learningRate)
	{
		for (int neuron = 0; neuron < neuronCount; neuron++)
		{
			// Update the weights
			for (int input = 0; input < inputCount; input++)
			{
				weights[neuron][input] -= learningRate * inputs[input] * errors[neuron];
			}

			// Update the biases - treated as it would be a weight, with 1 as input
			biases[neuron] -= learningRate * errors[neuron];
		}
	}

	public float[][] getWeights()
	{
		return weights;
	}

	public float[] getBiases()
	{
		return biases;
	}
}
