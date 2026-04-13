from tinynn import MLP, Number

def main():
    X = [
        [-1.0, -1.0],
        [-1.0,  1.0],
        [ 1.0, -1.0],
        [ 1.0,  1.0]
    ]
    Y = [-1.0, 1.0, 1.0, -1.0]

    model = MLP(
        layer_sizes=[2, 4, 4, 1], 
        act_fn=lambda x: x.tanh(), 
        seed=42
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
            print(f"Epoch {epoch:3d} | Toplam Hata (Loss): {total_loss.val:.4f}")

    print("\n🎯 Eğitim Bitti! Test Sonuçları:")
    for x_val, y_val in zip(X, Y):
        inputs = [Number(xi) for xi in x_val]
        pred = model(inputs)

        durum = "BAŞARILI" if (pred.val > 0) == (y_val > 0) else "BAŞARISIZ"

        print(f"Girdi: {x_val} | Beklenen: {y_val:4.1f} | Tahmin: {pred.val:5.2f} -> {durum}")


if __name__ == "__main__":
    main()
