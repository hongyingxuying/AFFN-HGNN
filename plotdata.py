import matplotlib.pyplot as plt

# 从训练日志文件中读取信息
log_file_path = 'checkpoint/Node_GAT_PUPath_FD_1110-232851/train.log'
epochs = []
train_losses = []
train_accs = []
val_losses = []
val_accs = []

with open(log_file_path, 'r') as file:
    lines = file.readlines()

for line in lines:
    if 'Epoch:' in line:
        # 解析 Epoch、Train Loss 和 Train Acc
        parts = line.split(',')
        epoch_info = parts[0].split()
        epochs.append(int(epoch_info[1]))
        train_loss = float(parts[1].split()[3])
        train_acc = float(parts[2].split()[2][:-1])  # Removing the comma at the end
        train_losses.append(train_loss)
        train_accs.append(train_acc)

    elif 'val-Loss' in line:
        # 解析 val-Loss 和 val-Acc
        parts = line.split()
        val_loss = float(parts[2])
        val_acc = float(parts[4][:-1])  # Removing the comma at the end
        val_losses.append(val_loss)
        val_accs.append(val_acc)

# 绘制损失图像
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses, label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 绘制准确度图像
plt.plot(epochs, train_accs, label='Train Accuracy')
plt.plot(epochs, val_accs, label='Validation Accuracy')
plt.title('Training and Validation Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
