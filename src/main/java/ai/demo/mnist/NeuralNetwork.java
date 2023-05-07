package ai.demo.mnist;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork
{
	private final List<NeuronLayer> neuronLayers = new ArrayList<>();
	private final float learningRate;

	public NeuralNetwork(float learningRate)
	{
		this.learningRate = learningRate;
	}

	public void addNeuronLayer(NeuronLayer neuronLayer)
	{
		this.neuronLayers.add(neuronLayer);
	}

	public void train(float[] input, float[] target)
	{
		// Feed forward
		float[] output = feedForward(input, true);

		// Calculate error on the output layer
		float[] errors = new float[output.length];
		for (int i = 0; i < output.length; i++)
		{
			errors[i] = output[i] - target[i];
		}

		// Back-propagate error to the previous layers
		float[] prevErrors = errors;
		for (int i = neuronLayers.size() - 1; i >= 0; i--)
		{
			prevErrors = neuronLayers.get(i).backPropagateErrors(prevErrors);
		}

		// Update the parameters using the
		for (NeuronLayer neuronLayer : neuronLayers)
		{
			neuronLayer.updateParameters(learningRate);
		}
	}

	public int test(float[] input)
	{
		float[] outputs = feedForward(input, false);

		// Find the index of the highest output
		int maxIndex = 0;

		for (int i = 1; i < outputs.length; i++)
		{
			if (outputs[i] > outputs[maxIndex])
			{
				maxIndex = i;
			}
		}

		return maxIndex;
	}

	public float[] feedForward(float[] input, boolean isTraining)
	{
		float[] hiddenState = input;

		for (NeuronLayer neuronLayer : neuronLayers)
		{
			hiddenState = neuronLayer.feedForward(hiddenState, isTraining);
		}

		return hiddenState;
	}

	public List<NeuronLayer> getNeuronLayers()
	{
		return neuronLayers;
	}
}
