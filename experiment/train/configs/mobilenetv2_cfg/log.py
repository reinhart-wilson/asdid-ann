import matplotlib.pyplot as plt

def parse_log_file(file_path):
    history = {
        'loss': [],
        'accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    with open(file_path, 'r') as file:
        for line in file:
            if 'loss:' in line and 'accuracy:' in line:
                parts = line.split(' - ')
                for part in parts:
                    if 'val_loss:' in part:
                        val_loss = float(part.split('val_loss: ')[1].strip())
                        history['val_loss'].append(val_loss)
                    elif 'val_accuracy:' in part:
                        val_accuracy = float(part.split('val_accuracy: ')[1].strip())
                        history['val_accuracy'].append(val_accuracy)
                    elif 'loss:' in part:
                        loss = float(part.split('loss: ')[1].strip())
                        history['loss'].append(loss)
                    elif 'accuracy:' in part:
                        accuracy = float(part.split('accuracy: ')[1].strip())
                        history['accuracy'].append(accuracy)
                    
    
    return history

def plot_loss(history):
    # Plot training loss vs. validation loss
    plt.figure(figsize=(12, 4))
    
    # Plot Training Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(history['loss']) + 1), history['loss'], label='Training Loss')
    plt.plot(range(1, len(history['val_loss']) + 1), history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Validation Loss')
    plt.legend()
    
    # Menampilkan bilangan bulat pada sumbu x dengan interval yang sesuai
    epoch_range = range(1, len(history['loss']) + 1)
    plt.xticks(epoch_range[::max(len(epoch_range)//10, 1)], rotation=0)  # Interval setiap 10 epoch
    plt.grid()
    
    # Plot Training Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(history['accuracy']) + 1), history['accuracy'], label='Training Accuracy')
    plt.plot(range(1, len(history['val_accuracy']) + 1), history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy vs Validation Accuracy')
    plt.legend()
    
    # Menampilkan bilangan bulat pada sumbu x dengan interval yang sesuai
    plt.xticks(epoch_range[::max(len(epoch_range)//10, 1)], rotation=0)  # Interval setiap 10 epoch
    plt.grid()
    
    # Tampilkan plot
    plt.tight_layout()  # Agar subplot tidak tumpang tindih
    plt.show()

# Example usage
log_file_path = 'log.txt'
history = parse_log_file(log_file_path)
plot_loss(history)
