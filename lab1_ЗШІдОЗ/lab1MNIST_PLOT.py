import json
import matplotlib.pyplot as plt
import os

# Читаємо дані
with open('training_data/mnist_history.json', 'r') as f:
    data = json.load(f)

# Створюємо графік
plt.figure(figsize=(10, 6))

# Втрати
plt.plot(data['loss'], 'b-', label='Train Loss', linewidth=2)
plt.plot(data['val_loss'], 'r-', label='Validation Loss', linewidth=2)

# Налаштування
plt.title('MNIST Model Training', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# Зберігаємо PNG
plt.savefig('training_plot.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"✅ Графік збережено: {os.path.abspath('training_plot.png')}")
print(f"   Test accuracy: {data['test_accuracy']:.4f}")