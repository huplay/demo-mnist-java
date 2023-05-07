package ai.demo.mnist;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBuffer;
import java.awt.image.DataBufferByte;
import java.io.*;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;

public class FileUtil
{
    public static float[] readPngFile(File file) throws Exception
    {
        BufferedImage image = ImageIO.read(file);
        DataBuffer dataBuffer = image.getRaster().getDataBuffer();
        DataBufferByte dataBufferByte = (DataBufferByte) dataBuffer;

        byte[] bytes = dataBufferByte.getData();

        float scale = 0.99f / 255;

        float[] floats = new float[bytes.length];

        for (int i = 0; i < bytes.length; i++)
        {
            floats[i] = (float) (bytes[i] * scale + 0.01);
        }

        return floats;
    }

    public static void createWeightFile(String fileName, float[][] weights) throws Exception
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

    public static float[][] readWeightFile(String fileName, int neuronCount, int inputCount)
    {
        return splitVector(read(fileName, neuronCount * inputCount), neuronCount);
    }

    public static float[] readBiasFile(String fileName, int neuronCount)
    {
        return read(fileName, neuronCount);
    }

    private static float[] read(String fileName, int size)
    {
        File file = new File(fileName);

        float[] array = new float[size];

        try (FileInputStream stream = new FileInputStream(file))
        {
            FileChannel channel = stream.getChannel();
            ByteBuffer buffer = channel.map(FileChannel.MapMode.READ_ONLY, 0, (long) size * 4);
            //buffer.order(settings.getByteOrder());
            FloatBuffer floatBuffer = buffer.asFloatBuffer();

            floatBuffer.get(array, 0, size);
        }
        catch (Exception e)
        {
            throw new RuntimeException("Parameter file read error. (" + file.getName() + ")");
        }

        return array;
    }

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
