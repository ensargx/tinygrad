import random
from tinygrad import MLP, Number
from tinygrad.layer import Layer
import tinygrad.functional as F

def main():
    X = [
        [-1.0, -1.0],
        [-1.0,  1.0],
        [ 1.0, -1.0],
        [ 1.0,  1.0]
    ]
    Y = [-1.0, 1.0, 1.0, -1.0]

    seed = 24
    rng = random.Random(seed)
    model = MLP(
        Layer(2, 4, act_fn=F.relu, rng=rng),
        Layer(4, 4, act_fn=F.relu, rng=rng),
        Layer(4, 1, act_fn=F.linear, rng=rng),
    )

    epochs = 500
    learning_rate = 0.05

    for epoch in range(epochs):

        total_loss = Number(0.0, requires_grad=False)

        for x_val, y_val in zip(X, Y):
            inputs = [Number(xi) for xi in x_val]
            target = Number(y_val)

            pred = model(inputs)

            loss = (pred - target) ** 2

            total_loss = total_loss + loss

        for p in model.parameters:
            p._grad = 0.0

        total_loss.backprop()

        for p in model.parameters:
            p.val -= learning_rate * p._grad

        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d} | Loss: {total_loss.val:.4f}")

    print("="*50)
    for x_val, y_val in zip(X, Y):
        inputs = [Number(xi) for xi in x_val]
        pred = model(inputs)

        # type-check
        if isinstance(pred, list):
            pred = pred[0]

        print(f"input: {x_val} | expected: {y_val:4.1f} | prediction: {pred.val:5.2f}")

if __name__ == "__main__":
    main()
