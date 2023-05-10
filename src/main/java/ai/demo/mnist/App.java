package ai.demo.mnist;

import java.util.*;

/**
 * The MNIST demo app
 */
public class App
{
	public static void main(String[] args) throws Exception
	{
		if (args != null && args.length > 0)
		{
			// Reading command line parameters
			String model = args[0];
			String parameters = null;
			boolean isTrain = true;

			if (args[0].equalsIgnoreCase("--testOnly"))
			{
				isTrain = false;
				model = args[1];

				if (args.length > 2) parameters = args[2];
				else throw new RuntimeException("Not enough parameters for test. Usage: run --test <modelFolder> <parametersFolder>");
			}
			else if (args.length > 1) parameters = args[1];

			// Execute the test and train
			new App().execute(model, parameters, isTrain);
		}
		else
		{
			throw new Exception("Not enough parameters. Usage: run <modelPath> [<parametersFolder>]");
		}
	}

	/**
	 * Creates a neural network (new or loaded from parameter files), performs an initial test,
	 * and optionally trains the network in multiple epochs
	 */
	private void execute(String model, String parameters, boolean isTrain) throws Exception
	{
		System.out.println("MNIST demo app.");
		System.out.println("Model: " + model + (parameters == null ? "" : "(" + parameters + ")"));

		// Read settings
		String modelPath = "models/" + model;
		Settings settings = new Settings(modelPath);

		// Build neural network
		NeuralNetwork neuralNetwork = NeuralNetwork.createNeuralNetwork(settings, modelPath, parameters);

		// Read test examples
		List<Example> testExamples = FileUtil.readTestExamples();

		// Test (measure the percentage of recognition on the test dataset)
		test(neuralNetwork, testExamples);

		if (isTrain)
		{
			// Training the network

			List<Example> trainExamples = FileUtil.readTrainExamples();

			while (true)
			{
				// Train an epoch
				trainEpoch(neuralNetwork, trainExamples);

				// Test (measure the percentage of recognition on the test dataset)
				test(neuralNetwork, testExamples);

				System.out.println("Do you want to continue the training with the next epoch? (Y/N) ");
				if (!"Y".equalsIgnoreCase(new Scanner(System.in).next())) break;
			}

			// Save the parameters
			FileUtil.saveParameters(neuralNetwork, modelPath);
		}
	}

	private void trainEpoch(NeuralNetwork neuralNetwork, List<Example> examples)
	{
		System.out.print("\nTraining... ");

		for (Example example : examples)
		{
			float[] target = new float[10];
			target[example.getLabel()] = 1;

			neuralNetwork.train(example.getPixels(), target);
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
			float[] output = neuralNetwork.feedForward(example.getPixels());
			int result = determineResult(output);

			if (result == example.getLabel()) success++;
		}

		// Print test statistics
		int successPercentage = 100 * success / examples.size();
		System.out.println(" Success: " + successPercentage + "%");
	}

	private int determineResult(float[] output)
	{
		int maxIndex = 0;

		for (int i = 1; i < output.length; i++)
		{
			if (output[i] > output[maxIndex])
			{
				maxIndex = i;
			}
		}

		return maxIndex;
	}
}
