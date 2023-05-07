package ai.demo.mnist;

import java.io.File;

public class AppTesting
{
	private Settings settings;

	public static void main(String[] args) throws Exception
	{
		if (args != null && args.length >= 3)
		{
			int testSize = Integer.parseInt(args[0]);
			String model = args[1];
			String parameters = args[2];

			new AppTesting().execute(testSize, model, parameters);
		}
		else
		{
			System.out.println("Not enough parameters");
		}
	}

	private void execute(int testSize, String model, String parametersFolder) throws Exception
	{
		String modelPath = "models/" + model;

		// Read model.properties file with settings
		this.settings = new Settings(modelPath);

		// Build the neural network with random weights and biases
		NeuralNetwork neuralNetwork = new NeuralNetwork(settings.getLearningRate());

		int inputCount = settings.getInputX() * settings.getInputY();

		for (int i = 0; i < settings.getHiddenLayers(); i++)
		{
			int neuronCount = settings.getHiddenLayerSizes().get(i);

			NeuronLayer neuronLayer = new NeuronLayer(inputCount, neuronCount);

			// Read parameters
			String prefix = modelPath + "/" + parametersFolder + "/layer." + i + ".";
			neuronLayer.setWeights(FileUtil.readWeightFile(prefix + "w.dat", neuronCount, inputCount));
			neuronLayer.setBiases(FileUtil.readBiasFile(prefix + "b.dat", neuronCount));

			neuralNetwork.addNeuronLayer(neuronLayer);

			inputCount = neuronCount;
		}

		int success = 0;

		// Test
		for (int i = 0; i < settings.getCategories().size(); i++)
		{
			String category = settings.getCategories().get(i);

			String[] fileNames = readTestFiles(category);

			for (int n = 0; n < testSize; n++)
			{
				File file = new File(settings.getInputSource() + "/testing/" + category + "/" + fileNames[n]);
				float[] input = FileUtil.readPngFile(file);

				int result = neuralNetwork.test(input);

				if (result == i) success++;
			}
		}

		// Print test statistics
		int testCases = settings.getCategories().size() * testSize;
		int successPercentage = 100 * success / testCases;

		System.out.println("Test cases: " + testCases);
		System.out.println("Success: " + success + " (" + successPercentage + "%)");
		System.out.println("Mistake: " + (testCases - success) + " (" + (100 - successPercentage) + "%)");
	}

	private String[] readTestFiles(String category)
	{
		return new File(settings.getInputSource() + "/testing/" + category).list();
	}
}
