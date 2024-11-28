import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

# Definisikan model
model = Sequential([
    Dense(2, input_shape=(2,), activation='relu', use_bias=False, 
          name="hidden_layer"),
    Dense(1, activation='linear', use_bias=False, name="output_layer")
])

# Set bobot layer pertama (hidden_layer)
hidden_weights = np.array([[0.2, 0.4],  # A -> C, A -> D
                           [-0.3, 0.1]])  # B -> C, B -> D
model.get_layer('hidden_layer').set_weights([hidden_weights])

# Set bobot layer kedua (output_layer)
output_weights = np.array([[0.5],  # C -> F
                           [-0.6]])  # D -> F
model.get_layer('output_layer').set_weights([output_weights])

# Tampilkan model summary untuk memastikan dimensi
model.summary()

# Data
input_data = np.array([
    [0.5, 1.5],  # Sampel 1
    [1.0, -1.0],  # Sampel 2
    [-0.5, 0.5]   # Sampel 3
])
target_data = np.array([
    [2.0],  # Target untuk sampel 1
    [1.0],  # Target untuk sampel 2
    [0.5]   # Target untuk sampel 3
])

# Kompilasi model
optimizer = SGD(learning_rate=0.1)
model.compile(optimizer=optimizer, loss='mse')

# Lihat keluaran
print('y_hat:\n', model.predict(input_data))

# Lihat MSE
history = model.fit(input_data, target_data, epochs=1, batch_size=3)
print(f'MSE: {history.history["loss"][0]}')

# Lihat bobot-bobot
print('Bobot-bobot pada lapisan hidden:\n', 
      model.get_layer("hidden_layer").get_weights()[0])
print('Bobot-bobot pada lapisan output:\n', 
      model.get_layer("output_layer").get_weights()[0])