package ai.demo.mnist;

/**
 * Represents a single example (image of a digit)
 */
public class Example
{
    // The number what can be seen on the image
    private final int label;

    // The pixels of the image (28 * 28 pixels, 8 bit grayscale values)
    private final float[] pixels;

    public Example(String line)
    {
        // Split the comma separated line
        String[] split = line.split(",");

        // Read the label (first value)
        label = Integer.parseInt(split[0]);

        // Read the pixels
        pixels = new float[28 * 28];
        for (int i = 1; i < split.length; i++)
        {
            pixels[i - 1] = (float) (Float.parseFloat(split[i]) / 255.0);
        }
    }

    public int getLabel()
    {
        return label;
    }

    public float[] getPixels()
    {
        return pixels;
    }
}