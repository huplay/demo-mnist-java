package ai.demo.mnist.activation;

import static java.lang.Math.*;

public class GELU implements Activation
{
    private static final float C = (float) sqrt(2f / PI);

    /**
     * Gaussian Error Linear Unit (GELU) cumulative distribution activation function (approximate implementation)
     * Original paper: <a href="https://paperswithcode.com/method/gelu" />
     */
    public float forward(float x)
    {
        return (float)(0.5 * x * (1 + tanh(C * (x + 0.044715 * x * x * x))));
    }

    /**
     * The derivative of GELU
     * https://arxiv.org/pdf/2104.02523.pdf
     */
    public float gradient(float x)
    {
        double a = 0.0356774 * x * x * x + 0.797885 * x;
        double c = cosh(a);

        return (float) ( (1 + tanh(a)) * 0.5 + (0.0535161 * x * x * x + 0.398942 * x) / c / c );
    }
}
