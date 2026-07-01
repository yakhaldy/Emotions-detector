
import os
import datetime
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D,
    Flatten, Dense, Dropout, BatchNormalization,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from preprocess import load_data, preprocess_pixels
from validation_loss_accuracy import plot_validation_loss_accuracy

print("\n==> Loading data...")
data = load_data("data/train.csv")
print(f"Total samples: {len(data)}")


print("\n==> Preprocessing images...")
X = preprocess_pixels(data["pixels"].values)
y = data["emotion"].to_numpy()
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")


print("\n==> Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)
print(f"Training samples : {len(X_train)}")
print(f"Test samples     : {len(X_test)}")


print("\n==> Computing class weights...")
class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train,
)
class_weights = dict(enumerate(class_weights_array))
print(f"Class weights: {class_weights}")


print("\n==> Setting up data augmentation...")
datagen = ImageDataGenerator(
    horizontal_flip=True,    
    rotation_range=10,      
    zoom_range=0.1,          
)
datagen.fit(X_train)


print("\n==> Building model...")
model = Sequential()

model.add(Input(shape=(48, 48, 1)))


model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), padding="same", activation="relu")) 
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.25))


model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), padding="same", activation="relu")) 
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.25))


model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding="same", activation="relu")) 
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))  
model.add(Dropout(0.25))

model.add(Flatten())                              
model.add(Dense(512, activation="relu"))           
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(256, activation="relu"))           
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(7, activation="softmax"))          

model.summary()


model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,              
    restore_best_weights=True,
    verbose=1,
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,               
    min_lr=1e-6,
    verbose=1,
)

LOG_DIR = os.path.join(
    "results", "logs",
    datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
)
tensorboard_cb = TensorBoard(
    log_dir=LOG_DIR,
    histogram_freq=1,
    write_graph=True,
    update_freq="epoch",
)

print(f"\nTensorBoard: tensorboard --logdir results/logs")


print("\n==> Training started...")
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_test, y_test),
    epochs=60,                 
    callbacks=[early_stop, reduce_lr, tensorboard_cb],
    class_weight=class_weights,
    verbose=1,
)


print("\n==> Evaluating...")
loss, acc = model.evaluate(X_test, y_test, verbose=1)
print(f"\nFinal Accuracy : {acc*100:.2f}%")
print(f"   Final Loss     : {loss:.4f}")


os.makedirs(os.path.join("results", "model"), exist_ok=True)
MODEL_PATH = os.path.join("results", "model", "final_emotion_model.keras")
model.save(MODEL_PATH)
print(f"\nModel saved → {MODEL_PATH}")


ARCH_PATH = os.path.join("results", "model", "final_emotion_model_arch.txt")
with open(ARCH_PATH, "w") as f:
    f.write("=== CNN v2 Architecture (CPU-friendly, target 60%+) ===\n\n")
    f.write("3 Convolutional Blocks avec double Conv:\n")
    f.write("  Bloc 1: Conv32 → Conv32 → MaxPool → Dropout(0.25)\n")
    f.write("  Bloc 2: Conv64 → Conv64 → MaxPool → Dropout(0.25)\n")
    f.write("  Bloc 3: Conv128 → Conv128 → MaxPool → Dropout(0.25)\n\n")
    f.write("Classification Head:\n")
    f.write("  Dense(512) → Dense(256) → Dense(7, softmax)\n\n")
    f.write("Choix d'architecture:\n")
    f.write("- Double Conv par bloc : extrait plus de features\n")
    f.write("- BatchNorm apres chaque Conv : training stable\n")
    f.write("- Dropout 0.25 conv + 0.5/0.3 head : evite overfitting\n")
    f.write("- Augmentation: flip + rotation(10) + zoom(0.1)\n")
    f.write("- Donnees completes (pas de reduction)\n")
    f.write("- patience=10 EarlyStopping, patience=4 ReduceLR\n\n")
    f.write("=== model.summary() ===\n\n")
    model.summary(print_fn=lambda line: f.write(line + "\n"))
    f.write(f"\n=== Results ===\n")
    f.write(f"Final test loss     : {loss:.4f}\n")
    f.write(f"Final test accuracy : {acc:.4f}\n")
print(f"Architecture saved → {ARCH_PATH}")


CURVES_PATH = os.path.join("results", "model", "learning_curves.png")
plot_validation_loss_accuracy(history, CURVES_PATH)

print("\nDone!")
print(f"   Model    : {MODEL_PATH}")
print(f"   Arch     : {ARCH_PATH}")
print(f"   Curves   : {CURVES_PATH}")
print(f"   TBoard   : {LOG_DIR}")