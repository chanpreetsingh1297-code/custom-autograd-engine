
import argparse
import numpy as np
import matplotlib.pyplot as plt
from custom_autograd.engine import Value, no_grad, FastCompiledTopology
from custom_autograd.nn import MLP
from custom_autograd.data import generate_data, split_data
from tqdm.auto import tqdm

# Plotting and visualizing the results
def plot_results(model_name, epochs: list, train_loss: list, test_loss: list, train_accuracy: list, test_accuracy: list):

  # 1. data sanitization
  unp_epochs = [int(e) for e in epochs]
  unp_train_loss = [float(tl.data) if hasattr(tl, 'data') else float(tl) for tl in train_loss]
  unp_test_loss = [float(vl.data) if hasattr(vl, 'data') else float(vl) for vl in test_loss]
  unp_train_acc = [float(ta.data) if hasattr(ta, 'data') else float(ta) for ta in train_accuracy]
  unp_test_acc = [float(va.data) if hasattr(va, 'data') else float(va) for va in test_accuracy]

  # 2. canvas setup
  plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

  # color palette
  COLOR_TRAIN_PRIMARY = "#1A5276"
  COLOR_TEST_SECONDARY = "#E67E22"
  COLOR_GRID = "#E5E8E8"

  # 3. subplot 1 - convergence loss trajectory
  ax1.plot(unp_epochs, unp_train_loss, label="Train", color=COLOR_TRAIN_PRIMARY, linewidth=1.75, alpha=0.95)
  ax1.plot(unp_epochs, unp_test_loss, label="Test / Val", color=COLOR_TEST_SECONDARY, linewidth=1.5, linestyle='--', alpha=0.9)

  ax1.set_title("Optimization Objective (Loss)", fontsize=12, fontweight='bold', pad=12, loc='left')
  ax1.set_xlabel("Optimization Epochs", fontsize=10, labelpad=8)
  ax1.set_ylabel("Mean Squared Error (MSE)", fontsize=10, labelpad=8)

  # structural refinements
  ax1.grid(True, linestyle='-', linewidth=0.5, color=COLOR_GRID)
  ax1.spines[['top', 'right']].set_visible(False)
  ax1.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=9, loc='upper right')

  # 4. subplot 2 - generalization accuracy evolution
  ax2.plot(unp_epochs, unp_train_acc, label="Train", color=COLOR_TRAIN_PRIMARY, linewidth=1.75, alpha=0.95)
  ax2.plot(unp_epochs, unp_test_acc, label="Test / Val", color=COLOR_TEST_SECONDARY, linewidth=1.5, linestyle='--', alpha=0.9)

  # typography and labels
  ax2.set_title("Generalization Capacity (Accuracy)", fontsize=12, fontweight='bold', pad=12, loc='left')
  ax2.set_xlabel("Optimization Epochs", fontsize=10, labelpad=8)
  ax2.set_ylabel("Classification Accuracy (%)", fontsize=10, labelpad=8)
  ax2.set_ylim(-5, 105)

  # global canvas adjustments and save output
  plt.suptitle(f"Model Training Evaluation Report Summary ({model_name.upper()})",
                 fontsize=14, fontweight='semibold', color='#2C3E50', y=0.98)

  plt.tight_layout()

  # export to Workspace
  plt.savefig('metrics_curve.png', bbox_inches='tight', dpi=300)
  print("\n[INFO] Analytics dashboard saved successfully to file 'metrics_curve.png' [DPI=300]")
  plt.show()


def train_model(model, fast_compiled_topology, grid_logits, train_data, train_labels, batch_size, lr):
  '''
  Trains the model in small chunks of size batch_size for full dataset per epoch
  Args:
    model = model used for training
    fast_compiled_topology = static compiled topology generated for speed
    grid_logits = default static logits
    train_data = training dataset
    train_labels = training labels
    batch_size = batch_size per epoch for training model
    lr = learning rate of the model
  Examples:
    train_loss, train_acc = train_model(model=model, fast_compiled_topology=fast_compiled_topology, grid_logits=grid_logits,
                                        train_data=train_dataset, train_labels=train_labels, batch_size=16, lr=0.01)
  '''
  total_epoch_loss = 0.0
  total_correct_preds = 0
  total_samples = 0

  # shuffling the training data
  indices = np.arange(len(train_data))
  np.random.shuffle(indices)
  k = len(train_data) % batch_size
  indices = indices[:-k] if k!=0 else indices

  # training the model
  for i in range(0, len(indices), batch_size):
    batch_indices = indices[i:i+batch_size]
    data_batch = [train_data[index] for index in batch_indices]
    labels_batch = [train_labels[index] for index in batch_indices]

    # forward pass
    fast_compiled_topology.update_batch(data_batch, labels_batch)

    batch_loss = fast_compiled_topology.forward()
    total_epoch_loss = total_epoch_loss + batch_loss

    pred_labels = [1 if logit.data>=0.0 else -1 for logit in grid_logits]

    # backward pass
    for p in model.parameters():
      p.grad = 0.0

    fast_compiled_topology.backward(model.parameters())

    # update weights
    for p in model.parameters():
      p.data = p.data + (-lr/batch_size * p.grad)

    # correct preditions and total samples per batch
    total_correct_preds += sum([pred_label==true_label for pred_label, true_label in zip(pred_labels, labels_batch)])
    total_samples = total_samples + batch_size

  # loss nad accuracy calculations
  total_epoch_loss = total_epoch_loss / total_samples * batch_size
  accuracy = total_correct_preds * 100 / total_samples

  return total_epoch_loss, accuracy


def test_model(model, fast_compiled_topology, grid_logits, test_data, test_labels, batch_size):
  '''
  Tests the model's inference capibilities on test_data divided into chunks of batch_size per epoch
  Args:
    model = model used for training
    fast_compiled_topology = static compiled topology generated for speed
    grid_logits = default static logits
    train_data = training dataset
    train_labels = training labels
    batch_size = batch_size per epoch for training model
    lr = learning rate of the model
  Examples:
    train_loss, train_acc = test_model(model=model, fast_compiled_topology=fast_compiled_topology, grid_logits=grid_logits,
                                        test_data=test_dataset, test_labels=test_labels, batch_size=16)
  '''
  total_epoch_loss = 0.0
  total_correct_preds = 0
  total_samples = 0

  # getting the indices
  indices = np.arange(len(test_data))
  k = len(test_data) % batch_size
  indices = indices[:-k] if k!=0 else indices

  # testing the model
  for i in range(0, len(indices), batch_size):
    batch_indices = indices[i:i+batch_size]
    data_batch = [test_data[index] for index in batch_indices]
    labels_batch = [test_labels[index] for index in batch_indices]

    # forward pass
    fast_compiled_topology.update_batch(data_batch, labels_batch)
    batch_loss = fast_compiled_topology.forward()
    total_epoch_loss = total_epoch_loss + batch_loss

    # total correct predictions and total samples calculation per batch
    pred_labels = [1 if logit.data>=0.0 else -1 for logit in grid_logits]
    total_correct_preds += sum([pred_label==true_label for pred_label, true_label in zip(pred_labels, labels_batch)])
    total_samples = total_samples + batch_size

  # returning total loss and accuracy
  total_epoch_loss = total_epoch_loss / total_samples * batch_size
  accuracy = total_correct_preds * 100 / total_samples
  return total_epoch_loss, accuracy


# Training Custom Binary Classification Model
def main():
  # 1. parser for command line interface
  parser = argparse.ArgumentParser(description="Train an OO-Autograd Multi-Layer Perceptron on a Hyperplane Classification Task.")

  # getting the hyperparameters
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

  BATCH_SIZE = 16

  epochs = args.epochs
  lr = args.lr
  lr_decay = args.lr_decay
  log_interval = 1 if epochs <= 500 else (epochs//100)

  static_graph_cache = None

  # model metrics storage
  epochs_list = []
  train_loss_list = []
  test_loss_list = []
  train_acc_list = []
  test_acc_list = []

  print(f"Initializing (Samples={NUM_SAMPLES}, Epochs={epochs}, Initial learning rate={lr})")

  # 1. generating the data
  data, labels = generate_data(num_samples=NUM_SAMPLES, num_features=NUM_FEATURES, low=low, high=high, random_seed=RANDOM_SEED)

  # 2. split data into training and testing datasets
  train_data, train_labels, test_data, test_labels = split_data(data=data, labels=labels, split_value=SPLIT_VALUE)

  # memory grid
  memory_grid, target_grid = [], []
  for _ in range(BATCH_SIZE):
    memory_grid.append([Value(0.0) for _ in range(NUM_FEATURES)])
    target_grid.append(Value(0.0))

  # 3. initialize the model
  # model = MLP(NUM_FEATURES, [4, 4, 1])
  model = MLP(NUM_FEATURES, [2, 1])

  # static computation initialization
  grid_logits = [model(x) for x in memory_grid]
  grid_squared_errors = [(yout-ygt)**2 for ygt, yout in zip(target_grid, grid_logits)]

  grid_loss = grid_squared_errors[0]
  for gl in grid_squared_errors[1:]:
    grid_loss = grid_loss + gl

  # initializing prebuilt graph using FastCompiledTopology class
  graph = FastCompiledTopology(loss_node=grid_loss, input_placeholders=memory_grid, target_placeholders=target_grid)

  # training and testing the model
  for epoch in tqdm(range(epochs)):
    # training the model
    train_loss, train_acc = train_model(model=model, fast_compiled_topology=graph, grid_logits=grid_logits,
                                        train_data=train_data, train_labels=train_labels, batch_size=BATCH_SIZE, lr=lr)

    # testing the model
    test_loss, test_acc = 0.0, 0.0
    with no_grad():
      test_loss, test_acc = test_model(model=model, fast_compiled_topology=graph, grid_logits=grid_logits,
                                       test_data=test_data, test_labels=test_labels, batch_size=BATCH_SIZE)

    # learning rate adjustments for adaptive learning rate
    lr = lr * lr_decay

    # storing model's metrics data
    if epoch % log_interval == 0 or epochs == epoch-1:
      epochs_list.append(epoch)
      train_loss_list.append(train_loss)
      test_loss_list.append(test_loss)

      train_acc_list.append(train_acc)
      test_acc_list.append(test_acc)

    # display output
    if epoch%10 == 0 or epoch == epochs-1:
      sample_grad = model.parameters()[0].grad
      print(f"Epoch {epoch} | Train loss: {train_loss:.4f} | Train Accuracy: {train_acc:.1f}% | Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.1f}%")

  # visualizing the results
  plot_results(model_name="model_0", epochs=epochs_list, train_loss=train_loss_list, test_loss=test_loss_list, train_accuracy=train_acc_list, test_accuracy=test_acc_list)


if __name__ == "__main__":
  main()
