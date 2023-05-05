package ai.backpropagation;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.*;

public class AppTraining
{
	private Settings settings;

	public static void main(String[] args) throws Exception
	{
		if (args != null && args.length >= 2)
		{
			int trainSize = Integer.parseInt(args[0]);
			String model = args[1];
			String base = args.length > 2 ? args[2] : null;

			new AppTraining().execute(trainSize, model, base);
		}
		else
		{
			System.out.println("Not enough parameters");

			// TODO: Just for test:
			new AppTraining().execute(1000, "test", null);
		}
	}

	private void execute(int trainSize, String model, String base) throws Exception
	{
		// Read settings
		String modelPath = "models/" + model;
		this.settings = new Settings(modelPath);
		List<String[]> fileNames = readTrainFiles();

		// Build neural network
		NeuralNetwork neuralNetwork = buildNeuralNetwork(modelPath, base);

		// Test (without training)
		test(neuralNetwork, 100);

		while (true)
		{
			// Train an epoch
			train(neuralNetwork, trainSize, fileNames);

			// Test
			test(neuralNetwork, 100);

			System.out.println("Do you want to continue the training with the next epoch? (Y/N) ");
			if (! "Y".equalsIgnoreCase(new Scanner(System.in).next())) break;
		}

		// Save the parameters
		save(neuralNetwork, modelPath);
	}

	private NeuralNetwork buildNeuralNetwork(String modelPath, String base)
	{
		NeuralNetwork neuralNetwork = new NeuralNetwork(settings.getLearningRate());

		int inputCount = settings.getInputX() * settings.getInputY();

		for (int i = 0; i < settings.getHiddenLayers(); i++)
		{
			int neuronCount = settings.getHiddenLayerSizes().get(i);

			NeuronLayer neuronLayer = new NeuronLayer(inputCount, neuronCount);

			if (base == null)
			{
				neuronLayer.setWeights(generateRandomWeights(inputCount, neuronCount));
				neuronLayer.setBiases(generateRandomBiases(neuronCount));
			}
			else
			{
				// Read parameters
				String prefix = modelPath + "/" + base + "/layer." + i + ".";
				neuronLayer.setWeights(FileUtil.readWeightFile(prefix + "w.dat", neuronCount, inputCount));
				neuronLayer.setBiases(FileUtil.readBiasFile(prefix + "b.dat", neuronCount));
			}

			neuralNetwork.addNeuronLayer(neuronLayer);

			inputCount = neuronCount;
		}

		return neuralNetwork;
	}

	private void train(NeuralNetwork neuralNetwork, int trainSize, List<String[]> fileNames) throws Exception
	{
		for (int i = 0; i < trainSize; i++)
		{
			File file = pickRandomTrainingImage(fileNames);

			float[] input = FileUtil.readPngFile(file);

			String[] path = file.getParent().split("\\\\");

			float[] expectedOutput = getOneHotArray(path[path.length - 1]);

			neuralNetwork.train(input, expectedOutput);
		}
	}

	private void test(NeuralNetwork neuralNetwork, int testSize) throws Exception
	{
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

	private void save(NeuralNetwork neuralNetwork, String modelPath) throws Exception
	{
		String folder = modelPath + "/parameters-" + getSysdate();
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
	}

	private String getSysdate()
	{
		String DATE_FORMAT_NOW = "yyyy-MM-dd-HH-mm-ss";

		Calendar cal = Calendar.getInstance();
		SimpleDateFormat format = new SimpleDateFormat(DATE_FORMAT_NOW);
		return format.format(cal.getTime());
	}

	private float[][] generateRandomWeights(int inputCount, int neuronCount)
	{
		float[][] randomWeights = new float[neuronCount][inputCount];

		for (int n = 0; n < neuronCount; n++)
		{
			for (int x = 0; x < inputCount; x++)
			{
				randomWeights[n][x] = (float) (Math.random() - 0.5);
			}
		}

		return randomWeights;
	}

	private float[] generateRandomBiases(int neuronCount)
	{
		float[] randomBiases = new float[neuronCount];

		for (int n = 0; n < neuronCount; n++)
		{
			randomBiases[n] = (float) (Math.random() - 0.5);
		}

		return randomBiases;
	}

	private List<String[]> readTrainFiles()
	{
		List<String> categories = settings.getCategories();
		List<String[]> fileNames = new ArrayList<>(categories.size());

		for (String category : categories)
		{
			fileNames.add(new File(settings.getInputSource() + "/training/" + category).list());
		}

		return fileNames;
	}

	private String[] readTestFiles(String category)
	{
		return new File(settings.getInputSource() + "/testing/" + category).list();
	}

	private File pickRandomTrainingImage(List<String[]> fileNames)
	{
		int randomCategoryIndex = new Random().nextInt(settings.getCategoryCount());
		String randomCategory = settings.getCategories().get(randomCategoryIndex);
		String[] randomCategoryFiles = fileNames.get(randomCategoryIndex);

		int randomFileIndex = new Random().nextInt(randomCategoryFiles.length);
		String randomFileName = randomCategoryFiles[randomFileIndex];

		return new File(settings.getInputSource() + "/training/" + randomCategory + "/" + randomFileName);
	}

	private float[] getOneHotArray(String value)
	{
		int index = settings.getCategoryMap().get(value);

		float[] oneHotArray = new float[settings.getCategoryCount()];

		for (int i = 0; i < settings.getCategoryCount(); i++)
		{
			oneHotArray[i] = (i == index) ? 0.99f : 0.01f;
		}

		return oneHotArray;
	}
}
