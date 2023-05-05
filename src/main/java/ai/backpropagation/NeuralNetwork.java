package ai.backpropagation;

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

	public void train(float[] input, float[] expectedOutput)
	{
		// Feed forward
		float[] output = feedForward(input, true);

		// Calculate difference
		float[] difference = getDifference(output, expectedOutput);

		// Back-propagate error
		for (int i = neuronLayers.size() - 1; i >= 0; i--)
		{
			NeuronLayer neuronLayer = neuronLayers.get(i);

			difference = neuronLayer.backPropagate(difference);
		}

		// Update weights and biases
		for (NeuronLayer neuronLayer : neuronLayers)
		{
			neuronLayer.update(learningRate);
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

	public float[] getDifference(float[] output, float[] expectedOutput)
	{
		float[] difference = new float[output.length];

		for (int i = 0; i < output.length; i++)
		{
			difference[i] = output[i] - expectedOutput[i];
		}

		return difference;
	}

	public List<NeuronLayer> getNeuronLayers()
	{
		return neuronLayers;
	}
}
