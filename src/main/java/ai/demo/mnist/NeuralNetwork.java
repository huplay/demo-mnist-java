package ai.demo.mnist;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Represents a neural network
 */
public class NeuralNetwork
{
	private final List<NeuronLayer> neuronLayers;
	private final float learningRate;

	/**
	 * Constructor
	 */
	public NeuralNetwork(List<NeuronLayer> neuronLayers, float learningRate)
	{
		this.neuronLayers = neuronLayers;
		this.learningRate = learningRate;
	}

	/**
	 * Creates a neural network based on the configuration
	 * The parameters will be initialized randomly, or using the stored parameters (in files)
	 */
	public static NeuralNetwork createNeuralNetwork(Settings settings, String modelPath, String parametersFolder)
	{
		List<NeuronLayer> neuronLayers = new ArrayList<>(settings.getLayerSizes().size());

		int inputCount = 28 * 28;
		for (int i = 0; i < settings.getLayerCount(); i++)
		{
			int neuronCount = settings.getLayerSizes().get(i);

			float[][] weights;
			float[] biases;

			if (parametersFolder == null)
			{
				// If this is a new model to be trained, generate random parameters
				Random rnd = new Random();
				weights = generateRandomWeights(inputCount, neuronCount, rnd);
				biases = generateRandomBiases(neuronCount, rnd);
			}
			else
			{
				// If this is an already trained model, read parameters from files
				String prefix = modelPath + "/" + parametersFolder + "/layer." + i + ".";
				weights = FileUtil.readWeightFile(prefix + "w.dat", neuronCount, inputCount);
				biases = FileUtil.readBiasFile(prefix + "b.dat", neuronCount);
			}

			neuronLayers.add(new NeuronLayer(inputCount, neuronCount, weights, biases, settings.getActivation()));

			inputCount = neuronCount;
		}

		if (parametersFolder == null) System.out.println("Parameters are initialized randomly.");
		else System.out.println("Parameters are initialized from folder: " + parametersFolder);

		return new NeuralNetwork(neuronLayers, settings.getLearningRate());
	}

	/**
	 * Trains the neural network on a single example using back-propagation
	 * @param input the inputs of the example
	 * @param target the targeted output
	 */
	public void train(float[] input, float[] target)
	{
		// Feed forward step, getting the output of the last layer
		// Meanwhile the input and output will be stored at every layers
		float[] output = feedForward(input);

		// Calculate the errors on the output layer
		float[] errors = new float[output.length];
		for (int i = 0; i < output.length; i++)
		{
			errors[i] = output[i] - target[i];
		}

		// Back-propagate the errors to the previous layers and update the parameters
		float[] prevErrors = errors;
		for (int i = neuronLayers.size() - 1; i >= 0; i--)
		{
			NeuronLayer neuronLayer = neuronLayers.get(i);

			// Back-propagate errors
			prevErrors = neuronLayer.backPropagateErrors(prevErrors);

			// Update parameters
			neuronLayer.updateParameters(learningRate);
		}
	}

	/**
	 * Calculates the outputs of the neural network given the inputs
	 * and stores the inputs and outputs at every layers
	 * @param inputs the inputs
	 * @return the outputs
	 */
	public float[] feedForward(float[] inputs)
	{
		// Feed forward
		float[] hiddenState = inputs;

		for (NeuronLayer neuronLayer : neuronLayers)
		{
			hiddenState = neuronLayer.feedForward(hiddenState);
		}

		// Find the index of the highest output
		return hiddenState;
	}

	/**
	 * Generates random weights (between -1 and 1)
	 */
	private static float[][] generateRandomWeights(int inputCount, int neuronCount, Random rnd)
	{
		float[][] randomWeights = new float[neuronCount][inputCount];

		for (int n = 0; n < neuronCount; n++)
		{
			for (int x = 0; x < inputCount; x++)
			{
				randomWeights[n][x] = 2 * rnd.nextFloat() - 1;
			}
		}

		return randomWeights;
	}

	/**
	 * Generates random biases (between -1 and 1)
	 */
	private static float[] generateRandomBiases(int neuronCount, Random rnd)
	{
		float[] randomBiases = new float[neuronCount];

		for (int n = 0; n < neuronCount; n++)
		{
			randomBiases[n] = 2 * rnd.nextFloat() - 1;
		}

		return randomBiases;
	}

	public List<NeuronLayer> getNeuronLayers()
	{
		return neuronLayers;
	}
}
