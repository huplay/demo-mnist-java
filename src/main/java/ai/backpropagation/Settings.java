package ai.backpropagation;

import java.io.File;
import java.io.IOException;
import java.util.*;

public class Settings
{
    private final String  inputSource;
    private final int inputX;
    private final int inputY;
    private final int categoryCount;
    private final List<String> categories;
    private final Map<String, Integer> categoryMap;
    private final int hiddenLayers;
    private final List<Integer> hiddenLayerSizes;
    private final float learningRate;

    public Settings(String modelPath) throws Exception
    {
        // Read all properties from the model.properties file
        Map<String, String> properties = readProperties(modelPath + "/model.properties");

        inputSource = properties.get("input.source");
        inputX = toInt(properties.get("input.x"));
        inputY = toInt(properties.get("input.y"));

        categoryCount = toInt(properties.get("category.count"));
        categories = new ArrayList<>(categoryCount);
        categoryMap = new HashMap<>(categoryCount);
        for (int i = 0; i < categoryCount; i++)
        {
            String category = properties.get("category." + i);
            categories.add(category);
            categoryMap.put(category, i);
        }

        hiddenLayers = toInt(properties.get("hidden.layers"));
        hiddenLayerSizes = new ArrayList<>(hiddenLayers);
        for (int i = 0; i < hiddenLayers; i++)
        {
            hiddenLayerSizes.add(toInt(properties.get("hidden." + i + ".size")));
        }

        learningRate = toFloat(properties.get("learning.rate"));
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

    public String getInputSource()
    {
        return inputSource;
    }

    public int getInputX()
    {
        return inputX;
    }

    public int getInputY()
    {
        return inputY;
    }

    public int getCategoryCount()
    {
        return categoryCount;
    }

    public List<String> getCategories()
    {
        return categories;
    }

    public Map<String, Integer> getCategoryMap()
    {
        return categoryMap;
    }

    public int getHiddenLayers()
    {
        return hiddenLayers;
    }

    public List<Integer> getHiddenLayerSizes()
    {
        return hiddenLayerSizes;
    }

    public float getLearningRate()
    {
        return learningRate;
    }
}
