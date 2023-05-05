package ai.backpropagation;

import static java.lang.Math.*;

public class NeuronLayer
{
	private static final float C = (float) sqrt(2f / PI);

	private final int inputCount;
	private final int neuronCount;

	private float[][] weights;
	private float[] biases;

	private float[] inputs;
	private float[] outputs;
	
	private float[] deltas;

	public NeuronLayer(int inputCount, int neuronCount)
	{
		this.inputCount = inputCount;
		this.neuronCount = neuronCount;
	}

	/**
	 * Calculate output of a neural layer - apply weights, bias and activation function
 	 */
	public float[] feedForward(float[] inputs, boolean isTraining)
	{
		if (isTraining) this.inputs = inputs; // Save inputs (only at training)

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
			outputs[neuron] = activation(outputs[neuron]);
		}

		if (isTraining) this.outputs = outputs; // Save outputs (only at training)

		return outputs;
	}

	public float[] backPropagate(float[] outputErrors)
	{
		deltas = new float[neuronCount];
		for (int neuron = 0; neuron < neuronCount; neuron++)
		{
			deltas[neuron] = gradient(outputs[neuron]) * outputErrors[neuron];
		}

		float[] inputErrors = new float[inputCount];
		for (int input = 0; input < inputCount; input++)
		{
			for (int neuron = 0; neuron < neuronCount; neuron++)
			{
				inputErrors[input] += weights[neuron][input] * deltas[neuron];
			}
		}

		return inputErrors;
	}

	public void update(double learningRate)
	{
		for (int neuron = 0; neuron < neuronCount; neuron++)
		{
			for (int input = 0; input < inputCount; input++)
			{
				weights[neuron][input] -= learningRate * inputs[input] * deltas[neuron];
			}

			biases[neuron] -= learningRate * deltas[neuron]; // The bias is updated as it would be a weight, with 1 as input
		}
	}

	public float activation(float x)
	{
		return sigmoid(x);
	}

	public float gradient(float x)
	{
		return sigmoidGradient(x);
	}

	public float sigmoid(float x)
	{
		return (float) (1 / (1 + Math.exp(-x)));
	}

	public float sigmoidGradient(float x)
	{
		return x * (1 - x);
	}

	/**
	 * Gaussian Error Linear Unit (GELU) cumulative distribution activation function (approximate implementation)
	 * Original paper: <a href="https://paperswithcode.com/method/gelu" />
	 */
	public float gelu(double x)
	{
		return (float)(0.5 * x * (1 + tanh(C * (x + 0.044715 * x * x * x))));
	}

	/**
	 * The derivative of GELU
	 */
	public float geluGradient(float x)
	{
		// TODO: double-check is it correct
		double a = tanh(C * x + C * 0.044715 * x * x * x);
		return (float) (0.5 * (1 + a + x * (1 - a * a) * (C + C * 0.134145 * x * x)));
	}

	public float[][] getWeights()
	{
		return weights;
	}

	public void setWeights(float[][] weights)
	{
		this.weights = weights;
	}

	public float[] getBiases()
	{
		return biases;
	}

	public void setBiases(float[] biases)
	{
		this.biases = biases;
	}
}
