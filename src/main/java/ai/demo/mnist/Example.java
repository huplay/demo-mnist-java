package ai.demo.mnist;

public class Example
{
    private final int label;
    private final float[] data;

    public Example(String line)
    {
        String[] split = line.split(",");

        this.label = Integer.parseInt(split[0]);
        this.data = new float[28 * 28];

        for (int i = 1; i < split.length; i++)
        {
            this.data[i - 1] = (float) (Float.parseFloat(split[i]) / 255.0);
        }
    }

    public int getLabel()
    {
        return this.label;
    }

    public float[] getData()
    {
        return this.data;
    }
}