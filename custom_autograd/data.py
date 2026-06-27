import random

# Generating synthetic Binary Classification data and splits the data into training and testing datasets

def generate_data(num_samples: int=100, num_features: int=3, low: float=-2.0, high: float=2.0, random_seed: int=None):
  '''
    generates synthetic binary classification data for the model
    Args:
      num_samples = number of samples(default: 100),
      num_features = number of features(default: 3),
      low = float value representing lower bound for data(default: -2.0)
      high = float value representing higher bound for data(default: 2.0)
      random_seed = integer value to set the random seed(default: None)

    Example:
      generate_data(num_samples=1000, num_features=5, low=-2.0, high=2.0, random_seed=42)
  '''
  if random_seed:
    random.seed(42)

  data = []
  labels = []
  for sample in range(num_samples):
    row = [random.uniform(low, high) for _ in range(num_features)]
    data.append(row)

    if sum(row) > 0.0:
      labels.append(1)
    else:
      labels.append(-1)

  return data, labels

def split_data(data: list=[], labels: list= [], split_value: float=0.8):
  '''
    Splits the data into training and testing datasets and returns them
    Args:
      data = list containing data
      split_value = float value to split data(0.0 to 1.0)

    Example:
      split_data(data=data_list, split_value=0.8)
  '''
  if split_value < 0.0 or split_value > 1.0:
    raise Exception("split_value must be between 0.0 and 1.0")
    return

  n = int(split_value*len(data))
  train_data, train_labels = data[:n], labels[:n]
  test_data, test_labels = data[n:], labels[n:]

  return train_data, train_labels, test_data, test_labels
