package ai.demo.mnist.activation;

public class Sigmoid implements Activation
{
    public float forward(float x)
    {
        return (float) (1.0 / (1.0 + Math.exp(-x)));
    }

    public float gradient(float x)
    {
        return x * (1 - x);
    }
}
