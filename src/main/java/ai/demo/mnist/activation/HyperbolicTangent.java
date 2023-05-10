package ai.demo.mnist.activation;

import static java.lang.Math.tanh;

public class HyperbolicTangent implements Activation
{
    public float forward(float x)
    {
        return (float) tanh(0.5 * x);
    }

    public float gradient(float x)
    {
        return (float) (0.5 * (1 - x * x));
    }
}
