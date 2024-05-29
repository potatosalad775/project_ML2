import os
import json
import matplotlib.pyplot as plt

# Directory containing the saved checkpoint files
checkpoint_dir = "Transformer/LoRA_hidden_size-32/checkpoint-1337500/"

# Path to the directory containing the trainer_state.json file
trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.json")

# Load the trainer state JSON file
with open(trainer_state_path, "r") as f:
    trainer_state = json.load(f)

# Extract log history from trainer state
log_history = trainer_state.get("log_history", [])

# Filter log history to include only entries with eval_loss
train_losses = [entry["loss"] for entry in log_history if "loss" in entry and str(entry.get("epoch", "")).endswith('0')]
eval_losses = [entry["eval_loss"] for entry in log_history if "eval_loss" in entry]
eval_accuracy = [entry["eval_accuracy"]["accuracy"] for entry in log_history if "eval_accuracy" in entry]

# Plot the test loss values by epoch
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
plt.plot(range(1, len(eval_losses) + 1), eval_losses, label='Evaluation Loss')
plt.plot(range(1, len(eval_accuracy) + 1), eval_accuracy, label='Evaluation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Test Loss')
plt.title('Test Loss by Epoch')
plt.legend()
plt.show()