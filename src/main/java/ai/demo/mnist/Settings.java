package ai.demo.mnist;

import ai.demo.mnist.activation.Activation;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * To read and store all information of the model.properties file
 */
public class Settings
{
    private final List<Integer> layerSizes;
    private final String activation;
    private final float learningRate;

    public Settings(String modelPath) throws Exception
    {
        // Read all properties from the model.properties file
        Map<String, String> properties = readProperties(modelPath + "/model.properties");

        int hiddenLayers = toInt(properties.get("hidden.layers"));
        layerSizes = new ArrayList<>(hiddenLayers + 1);
        for (int i = 0; i < hiddenLayers; i++)
        {
            layerSizes.add(toInt(properties.get("hidden." + i + ".size")));
        }
        layerSizes.add(10);

        activation = properties.get("activation");

        learningRate = toFloat(properties.get("learning.rate"));

        System.out.print("Layers: (784)");
        for (int size : layerSizes) System.out.print(" - " + size);
        System.out.println("\nActivation function: " + activation);
        System.out.println("Learning rate: " + learningRate + "\n");
    }

    public static Map<String, String> readProperties(String fileName) throws Exception
    {
        Map<String, String> properties = new HashMap<>();

        try (Scanner scanner = new Scanner(new File(fileName)))
        {
            while (scanner.hasNextLine())
            {
                String line = scanner.nextLine();
                if (line != null && !line.trim().equals("") && !line.startsWith("#"))
                {
                    String[] parts = line.split("=");
                    if (parts.length == 2)
                    {
                        properties.put(parts[0].trim(), parts[1].trim());
                    }
                    else
                    {
                        System.out.println("\nWARNING: Unrecognizable properties line: (" + fileName + "): " + line);
                    }
                }
            }
        }
        catch (IOException e)
        {
            throw new Exception("Cannot read model.properties file: " + fileName);
        }

        return properties;
    }

    private int toInt(String value) throws Exception
    {
        try
        {
            return Integer.parseInt(value);
        }
        catch (Exception e)
        {
            throw new Exception("The provided properties value can't be converted to integer (" + value + ").");
        }
    }

    private float toFloat(String value) throws Exception
    {
        try
        {
            return Float.parseFloat(value);
        }
        catch (Exception e)
        {
            throw new Exception("The provided properties value can't be converted to float (" + value + ").");
        }
    }

    public int getLayerCount()
    {
        return layerSizes.size();
    }

    public List<Integer> getLayerSizes()
    {
        return layerSizes;
    }

    public Activation getActivation()
    {
        return Activation.getInstance(activation);
    }

    public float getLearningRate()
    {
        return learningRate;
    }
}
