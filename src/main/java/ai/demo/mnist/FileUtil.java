package ai.demo.mnist;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.List;
import java.util.Scanner;

/**
 * File utility to read train/test files, and to read and save parameter files
 */
public class FileUtil
{
    /**
     * Reads the train examples
     * (The train examples are split into three files because of the GitHub upload limit)
     */
    public static List<Example> readTrainExamples() throws Exception
    {
        List<Example> examples = readExamples(new File("src/main/resources/mnist_train_0.csv"));
        examples.addAll(readExamples(new File("src/main/resources/mnist_train_1.csv")));
        examples.addAll(readExamples(new File("src/main/resources/mnist_train_2.csv")));

        return examples;
    }

    /**
     * Reads the test examples
     */
    public static List<Example> readTestExamples() throws Exception
    {
        return readExamples(new File("src/main/resources/mnist_test.csv"));
    }

    /**
     * Reads examples from a csv file
     */
    private static List<Example> readExamples(File file) throws Exception
    {
        List<Example> examples = new ArrayList<>();

        try {
            Scanner sc = new Scanner(file);

            while (sc.hasNextLine())
            {
                examples.add(new Example(sc.nextLine()));
            }

            sc.close();

        }
        catch (IOException e)
        {
            throw new Error("File reading error: " + file.getName());
        }

        return examples;
    }

    /**
     * Reads a weight file
     */
    public static float[][] readWeightFile(String fileName, int neuronCount, int inputCount)
    {
        return splitVector(read(fileName, neuronCount * inputCount), neuronCount);
    }

    /**
     * Reads a bias file
     */
    public static float[] readBiasFile(String fileName, int neuronCount)
    {
        return read(fileName, neuronCount);
    }

    /**
     * Reads values of a parameter file
     */
    private static float[] read(String fileName, int size)
    {
        File file = new File(fileName);

        float[] array = new float[size];

        try (FileInputStream stream = new FileInputStream(file))
        {
            FileChannel channel = stream.getChannel();
            ByteBuffer buffer = channel.map(FileChannel.MapMode.READ_ONLY, 0, (long) size * 4);
            FloatBuffer floatBuffer = buffer.asFloatBuffer();

            floatBuffer.get(array, 0, size);
        }
        catch (Exception e)
        {
            throw new RuntimeException("Parameter file read error. (" + file.getName() + ")");
        }

        return array;
    }

    /**
     * Saves the parameters of a trained network
     */
    public static void saveParameters(NeuralNetwork neuralNetwork, String modelPath) throws Exception
    {
        System.out.print("Saving parameters... ");

        String time = new SimpleDateFormat("yyyy-MMdd-HH-mm").format(Calendar.getInstance().getTime());
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

    /**
     * Creates a weight file
     */
    private static void createWeightFile(String fileName, float[][] weights) throws Exception
    {
        File file = new File(fileName);
        DataOutputStream output = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(file)));

        for (int n = 0; n < weights.length; n++)
        {
            for (int i = 0; i < weights[n].length; i++)
            {
                output.writeFloat(weights[n][i]);
            }
        }

        output.close();
    }

    /**
     * Creates a bias file
     */
    public static void createBiasFile(String fileName, float[] biases) throws Exception
    {
        File file = new File(fileName);
        DataOutputStream output = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(file)));

        for (int n = 0; n < biases.length; n++)
        {
            output.writeFloat(biases[n]);
        }

        output.close();
    }

    /**
     * Splits a vector of weights to a matrix
     */
    private static float[][] splitVector(float[] vector, int count)
    {
        int size = vector.length / count;
        float[][] ret = new float[count][size];

        int segment = 0;
        int col = 0;
        for (float value : vector)
        {
            ret[segment][col] = value;

            if (col == size - 1)
            {
                col = 0;
                segment++;
            }
            else col++;
        }

        return ret;
    }
}
