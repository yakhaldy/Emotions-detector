import os
import datetime
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D,
    Flatten, Dense, Dropout, BatchNormalization,GlobalAveragePooling2D,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from preprocess import load_data, preprocess_pixels
from validation_loss_accuracy import plot_validation_loss_accuracy

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model


# ===============================================================
# 2. Load Data 
# ===============================================================
print("\n==> Loading data...")
data = load_data("data/train.csv")
print(f"Total samples: {len(data)}")

# ===============================================================
# 3. Preprocess
# ===============================================================
print("\n==> Preprocessing images...")
X_gray = preprocess_pixels(data["pixels"].values)
y = data["emotion"].to_numpy()
print(f"X shape: {X_gray.shape}")
print(f"y shape: {y.shape}")


X = np.repeat(X_gray, 3, axis=-1)
print(f"X shape after RGB conversion: {X.shape}")  # (N, 48, 48, 3)

# ===============================================================
# 4. Split Data — 80% train, 20% test
# ===============================================================
print("\n==> Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)
print(f"Training samples : {len(X_train)}")
print(f"Test samples     : {len(X_test)}")


# ===============================================================
# 6. Transfer Learning — MobileNetV2
# ===============================================================
print("\n==> Loading MobileNetV2 base...")
base_model = MobileNetV2(
    input_shape=(48, 48, 3),
    include_top=False,        
    weights="imagenet",       
)

# ===============================================================
# 7. Build Model — Base + Custom Head
# ===============================================================
print("\n==> Building model...")

inputs = Input(shape=(48, 48, 3))

x = base_model(inputs, training=False)

x = GlobalAveragePooling2D()(x)         # (N, 7, 7, 1280) → (N, 1280)

# Custom Classification Head
x = Dense(256, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

x = Dense(128, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

outputs = Dense(7, activation="softmax")(x)  # 7 emotions

model = Model(inputs, outputs)
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




print("\n==> Phase 1: Training head only (base frozen)...")
history1 = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_test, y_test),
    epochs=20,                    
    callbacks=[early_stop, reduce_lr, tensorboard_cb],
    class_weight=class_weights,
    verbose=1,
)

loss1, acc1 = model.evaluate(X_test, y_test, verbose=0)
print(f"\nPhase 1 Accuracy: {acc1*100:.2f}%")

# ===============================================================
# 11. Fine-tuning — 30 layers of MobileNetV2
# ===============================================================

print("\n==> Phase 2: Fine-tuning last 30 layers...")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False       
for layer in base_model.layers[-30:]:
    layer.trainable = True         

model.compile(
    optimizer=Adam(learning_rate=1e-5),  
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

early_stop2 = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True,
    verbose=1,
)

reduce_lr2 = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1,
)

history2 = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_test, y_test),
    epochs=30,
    callbacks=[early_stop2, reduce_lr2, tensorboard_cb],
    class_weight=class_weights,
    verbose=1,
)


print("\n==> Final Evaluation...")
loss, acc = model.evaluate(X_test, y_test, verbose=1)
print(f"\n🎯 Final Accuracy : {acc*100:.2f}%")
print(f"   Final Loss     : {loss:.4f}")
