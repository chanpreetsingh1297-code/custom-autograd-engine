
import argparse
from custom_autograd.engine import no_grad
from custom_autograd.nn import MLP
from custom_autograd.data import generate_data, split_data
from tqdm.auto import tqdm

# Training Custom Binary Classification Model

def main():
  # 1. Parser for command line interface
  parser = argparse.ArgumentParser(description="Train an OO-Autograd Multi-Layer Perceptron on a Hyperplane Classification Task.")

  parser.add_argument("--samples", type=int, default=100, help="Total number of samples in the dataset")
  parser.add_argument("--features", type=int, default=3, help="Total number of features per sample")
  parser.add_argument("--low", type=float, default=-2.0, help="Lower bound for per sample value")
  parser.add_argument("--high", type=float, default=2.0, help="Upper bound for per sample value")
  parser.add_argument("--seed", type=int, default=42, help="Random seed anchor for execution reproducibility")
  parser.add_argument("--split", type=float, default=0.8, help="Train/Test dataset partition split ratio")
  parser.add_argument("--epochs", type=int, default=300, help="Number of optimization training epochs")
  parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate parameter")
  parser.add_argument("--lr_decay", type=float, default=0.99, help="Learning rate decay parameter")

  args = parser.parse_args()

  NUM_SAMPLES = args.samples
  NUM_FEATURES = args.features
  low, high = args.low, args.high
  RANDOM_SEED = args.seed

  SPLIT_VALUE = args.split

  epochs = args.epochs
  lr = args.lr
  lr_decay = args.lr_decay

  print(f"Initializing (Samples={NUM_SAMPLES}, Epochs={epochs}, Initial learning rate={lr})")

  # 1. Generating the data
  data, labels = generate_data(num_samples=NUM_SAMPLES, num_features=NUM_FEATURES, low=low, high=high, random_seed=RANDOM_SEED)

  # 2. Split data into training and testing datasets
  train_data, train_labels, test_data, test_labels = split_data(data=data, labels=labels, split_value=SPLIT_VALUE)

  # 3. Initialize the model
  model = MLP(NUM_FEATURES, [4, 4, 1])

  for epoch in tqdm(range(epochs)):
    # Forward pass
    y_logits = [model(x) for x in train_data]
    y_pred_labels = [1 if logit.data>=0.0 else -1 for logit in y_logits]

    squared_errors = [(yout-ygt)**2 for ygt, yout in zip(train_labels, y_logits)]

    loss = squared_errors[0]
    for l in squared_errors[1:]:
      loss = loss + l

    # Backward pass
    for p in model.parameters():
      p.grad = 0.0
    loss.backward()

    # Update weights
    for p in model.parameters():
      p.data = p.data + (-lr * p.grad)

    lr = lr * lr_decay

    train_loss_normalized = loss.data / len(train_data)
    total_correct = sum([pred_label==true_label for pred_label, true_label in zip(y_pred_labels, train_labels)])
    accuracy = total_correct * 100 / len(train_labels)

    with no_grad():
      pred_logits = [model(y) for y in test_data]
      pred_labels = [1 if logit.data>=0.0 else -1 for logit in pred_logits]

      squared_pred_errors = [(yout-ygt)**2 for ygt, yout in zip(test_labels, pred_logits)]

      test_loss = squared_pred_errors[0]
      for tl in squared_pred_errors[1:]:
        test_loss = test_loss + tl

      test_loss = test_loss.data / len(test_data)
      test_correct = sum((pred_label==test_label) for pred_label, test_label in zip(pred_labels, test_labels))
      test_accuracy = test_correct * 100/len(test_labels)


    if epoch%10 == 0 or epoch == epochs-1:
      sample_grad = model.parameters()[0].grad
      print(f"Epoch {epoch} | Train loss: {train_loss_normalized:.4f} | Train Accuracy: {accuracy:.1f}% | Sample Grad: {sample_grad:.6f} | Test loss: {test_loss:.4f} | Test accuracy: {test_accuracy:.1f}")

if __name__ == "__main__":
  main()
