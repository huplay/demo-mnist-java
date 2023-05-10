package ai.demo.mnist.activation;

public interface Activation
{
    float forward(float x);

    float gradient(float x);

    static Activation getInstance(String activation)
    {
        switch (activation.toUpperCase())
        {
            case "SIGMOID": return new Sigmoid();
            case "TANH": return new HyperbolicTangent();
            case "GELU": return new GELU();
        }

        throw new RuntimeException("Unknown activation function: " + activation);
    }
}
