import torch
import matplotlib.pyplot as plt

def get_accuracy(history_path):
    history = torch.load(history_path)
    train_error_history =  1 - history['train_accuracy_history'][-1] / 100  # calculate test error
    test_error_history = 1 - history['test_accuracy_history'][-1] / 100  # calculate test error
    return train_error_history, test_error_history

width_params = [1, 2, 4, 8, 16, 32, 64]
data_paths = [f"5-Layer-CNN/mcnn_cifar10_adam/noise-0.0_datasize-1.0_w_param-{p}/history.pth" for p in width_params]

train_errors = [get_accuracy(path)[0] for path in data_paths]
test_errors = [get_accuracy(path)[1] for path in data_paths]

plt.plot(width_params, train_errors, label='Train Error')
plt.plot(width_params, test_errors, label='Test Error')
plt.xlabel('CNN Width Parameter')
plt.ylabel('Error Rate')
plt.title('Error Rate vs CNN Width')
plt.legend()
plt.show()