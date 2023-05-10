package ai.demo.mnist;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.*;

public class App
{
	public static void main(String[] args) throws Exception
	{
		if (args != null && args.length > 0)
		{
			// Reading command line parameters
			String model = args[0];
			String parameters = null;
			boolean isTestOnly = false;

			if (args[0].equalsIgnoreCase("--test"))
			{
				isTestOnly = true;
				model = args[1];

				if (args.length > 2) parameters = args[2];
				else throw new RuntimeException("Not enough parameters for test. Usage: run --test <modelFolder> <parametersFolder>");
			}
			else if (args.length > 1) parameters = args[1];

			// Execute the test and train
			new App().execute(model, parameters, isTestOnly);
		}
		else
		{
			throw new Exception("Not enough parameters. Usage: run <modelPath> [<parametersFolder>]");
		}
	}

	private void execute(String model, String parameters, boolean isTestOnly) throws Exception
	{
		// Read settings
		String modelPath = "models/" + model;
		Settings settings = new Settings(modelPath);

		// Build neural network
		NeuralNetwork neuralNetwork = buildNeuralNetwork(settings, modelPath, parameters);

		// Read test examples
		List<Example> testExamples = FileUtil.readTestExamples();

		// Initial test
		test(neuralNetwork, testExamples);

		if (!isTestOnly)
		{
			List<Example> trainExamples = FileUtil.readTrainExamples();

			while (true)
			{
				// Train an epoch
				trainEpoch(neuralNetwork, trainExamples);

				// Test
				test(neuralNetwork, testExamples);

				System.out.println("Do you want to continue the training with the next epoch? (Y/N) ");
				if (!"Y".equalsIgnoreCase(new Scanner(System.in).next())) break;
			}

			// Save the parameters
			saveParameters(neuralNetwork, modelPath);
		}
	}

	private NeuralNetwork buildNeuralNetwork(Settings settings, String modelPath, String parametersFolder)
	{
		NeuralNetwork neuralNetwork = new NeuralNetwork(settings.getLearningRate());

		int inputCount = 28 * 28;

		for (int i = 0; i < settings.getLayerCount(); i++)
		{
			int neuronCount = settings.getLayerSizes().get(i);

			NeuronLayer neuronLayer = new NeuronLayer(inputCount, neuronCount);

			if (parametersFolder == null)
			{
				Random rnd = new Random();
				neuronLayer.setWeights(generateRandomWeights(inputCount, neuronCount, rnd));
				neuronLayer.setBiases(generateRandomBiases(neuronCount, rnd));
			}
			else
			{
				// Read parameters
				String prefix = modelPath + "/" + parametersFolder + "/layer." + i + ".";
				neuronLayer.setWeights(FileUtil.readWeightFile(prefix + "w.dat", neuronCount, inputCount));
				neuronLayer.setBiases(FileUtil.readBiasFile(prefix + "b.dat", neuronCount));
			}

			neuralNetwork.addNeuronLayer(neuronLayer);

			inputCount = neuronCount;
		}

		if (parametersFolder == null) System.out.println("Parameters are initialized randomly.");
		else System.out.println("Parameters are initialized from folder: " + parametersFolder);

		return neuralNetwork;
	}

	private void trainEpoch(NeuralNetwork neuralNetwork, List<Example> examples)
	{
		System.out.print("Training... ");

		for (Example example : examples)
		{
			float[] target = new float[10];
			target[example.getLabel()] = 1;

			neuralNetwork.train(example.getData(), target);
		}

		System.out.println("Done");
	}

	private void test(NeuralNetwork neuralNetwork, List<Example> examples)
	{
		System.out.print("Testing...");

		int success = 0;

		// Test
		for (Example example : examples)
		{
			int result = neuralNetwork.test(example.getData());

			if (result == example.getLabel()) success++;
		}

		// Print test statistics
		int successPercentage = 100 * success / examples.size();
		System.out.println(" Success: " + successPercentage + "%");
	}

	private void saveParameters(NeuralNetwork neuralNetwork, String modelPath) throws Exception
	{
		System.out.print("Saving parameters... ");

		String time = new SimpleDateFormat("yyyy-MM-dd-HH-mm-ss").format(Calendar.getInstance().getTime());
		String folder = modelPath + "/parameters-" + time;

		if (new File(folder).mkdirs())
		{
			for (int layer = 0; layer < neuralNetwork.getNeuronLayers().size(); layer++)
			{
				NeuronLayer neuralLayer = neuralNetwork.getNeuronLayers().get(layer);

				FileUtil.createWeightFile(folder + "/layer." + layer + ".w.dat", neuralLayer.getWeights());
				FileUtil.createBiasFile(folder + "/layer." + layer + ".b.dat", neuralLayer.getBiases());
			}
		}
		else
		{
			System.out.println("Parameter folder creation error");
		}

		System.out.println("Done (" + folder + ")");
	}

	private float[][] generateRandomWeights(int inputCount, int neuronCount, Random rnd)
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

	private float[] generateRandomBiases(int neuronCount, Random rnd)
	{
		float[] randomBiases = new float[neuronCount];

		for (int n = 0; n < neuronCount; n++)
		{
			randomBiases[n] = 2 * rnd.nextFloat() - 1;
		}

		return randomBiases;
	}
}
