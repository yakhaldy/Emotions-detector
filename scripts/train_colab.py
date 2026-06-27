

# ===============================================================
# Emotion Detection - CNN VGG-like (objectif 60%+ val_accuracy)
# ===============================================================
# Dataset : FER2013 (48x48 grayscale, 7 emotions)
# IdÃ©e   : architecture plus profonde + LR scheduler + class weights
# ===============================================================

# --- Imports ---------------------------------------------------------------
import numpy as np                                          # calculs sur tableaux (matrices d'images, labels)
import pandas as pd                                         # lecture du CSV FER2013
import matplotlib.pyplot as plt                             # generation du PNG des learning curves
import tensorflow as tf                                     # framework deep learning
from tensorflow.keras.models import Sequential              # modele "empile les couches dans l'ordre"
from tensorflow.keras.layers import (
    Input,                                                  # couche d'entree explicite (remplace input_shape, plus propre)
    Conv2D,                                                 # convolution 2D : extrait des motifs locaux (yeux, bouche...)
    MaxPooling2D,                                           # reduit la taille spatiale en gardant le max d'une fenetre
    Flatten,                                                # transforme la matrice 3D en vecteur 1D pour les Dense
    Dense,                                                  # couche pleinement connectee (chaque neurone voit tout)
    Dropout,                                                # eteint aleatoirement des neurones -> evite l'overfitting
    BatchNormalization,                                     # normalise les activations -> entrainement plus stable et rapide
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator   # data augmentation a la volee
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard  # callbacks pendant le training
from tensorflow.keras.optimizers import Adam                # optimiseur Adam (configurable pour fixer le learning rate)
from sklearn.model_selection import train_test_split        # split train/test reproductible
from sklearn.utils.class_weight import compute_class_weight # calcul automatique des poids pour classes desequilibrees
from preprocess import load_data , preprocess_pixels

# ===========================================================================
# 0. Verification du GPU
# ===========================================================================
# Si la liste est vide -> tu es sur CPU, va activer le GPU dans Runtime.
# Sinon tu verras [PhysicalDevice(name='/physical_device:GPU:0', ...)].
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print("GPU disponible :", gpus)
if not gpus:
    print("⚠️ Aucun GPU detecte ! Active-le : Runtime > Change runtime type > T4 GPU.")



print("==> Loading data...")
data = load_data("data/train.csv")                        
print(data.head())                                            


print("==> Processing images...")
X = preprocess_pixels(data["pixels"])                                     
y = data["emotion"].to_numpy()                             


print("==>Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,                                          
    random_state=42,                                         
    stratify=y,                                              
)


# ===========================================================================
# 4. Class weights (compense le desequilibre du dataset)
# ===========================================================================
# FER2013 est tres desequilibre : "happy" (~25%) vs "disgust" (~1.5%).
# Sans poids, le modele apprend a tout predire "happy" pour optimiser l'accuracy.
# compute_class_weight('balanced') donne plus de poids aux classes rares.
print("Computing class weights...")

class_weights_array = compute_class_weight(
    class_weight="balanced",                                 # formule : n_samples / (n_classes * count_classe)
    classes=np.unique(y_train),                              # liste des labels uniques [0,1,2,3,4,5,6]
    y=y_train,                                               # labels du training set seulement
)
class_weights = dict(enumerate(class_weights_array))         # Keras attend un dict {0: poids0, 1: poids1, ...}
print("   class_weights =", class_weights)                   # affiche pour verifier


# ===========================================================================
# 5. Data augmentation (genere des variations a la volee)
# ===========================================================================
# Augmentation = on cree des images legerement modifiees a chaque epoch.
# Le modele voit "plus de donnees" -> generalise mieux, overfitte moins.
# Valeurs reduites vs avant (15 au lieu de 20, 0.15 au lieu de 0.2) :
# 48x48 c'est petit, des transfos trop fortes detruisent les details du visage.
print("==> Creating augmented data...")

datagen = ImageDataGenerator(
    rotation_range=15,                                       # rotation aleatoire jusqu'a +/- 15 degres
    width_shift_range=0.15,                                  # decale horizontalement jusqu'a 15% de la largeur
    height_shift_range=0.15,                                 # decale verticalement jusqu'a 15% de la hauteur
    zoom_range=0.15,                                         # zoom in/out aleatoire jusqu'a +/- 15%
    horizontal_flip=True,                                    # miroir horizontal (un visage reste un visage en miroir)
)
datagen.fit(X_train)                                         # calcule les stats internes (utile si featurewise_*=True)


# ===========================================================================
# 6. Construction du modele (architecture VGG-like)
# ===========================================================================
# Principe VGG : empiler 2 Conv puis MaxPool, repeter en doublant les filtres.
# - "padding='same'" garde la meme taille spatiale apres la conv (evite de
#   perdre les pixels du bord, important pour des images 48x48).
# - BatchNorm apres chaque Conv -> stabilise et accelere.
# - Dropout 0.25 dans les blocs conv -> regularisation legere.
# ===========================================================================
print("==> Building model...")

model = Sequential()

# -- Couche d'entree explicite (remplace input_shape, evite le warning Keras) --
model.add(Input(shape=(48, 48, 1)))                          # 1 image = 48x48x1 (grayscale)

# -- Bloc 1 : 64 filtres -> apprend des motifs simples (bords, gradients) --
model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))   # 64 filtres 3x3, sortie 48x48x64
model.add(BatchNormalization())                                     # normalise les sorties du Conv
model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))   # 2eme conv -> raffine les motifs detectes
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))                          # divise la taille par 2 -> 24x24x64
model.add(Dropout(0.25))                                            # eteint 25% des neurones aleatoirement

# -- Bloc 2 : 128 filtres -> motifs plus complexes (yeux, sourcils...) --
model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))  # sortie 24x24x128
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))                          # -> 12x12x128
model.add(Dropout(0.25))

# -- Bloc 3 : 256 filtres -> parties du visage (bouche, nez, expressions) --
model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))  # sortie 12x12x256
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))                          # -> 6x6x256
model.add(Dropout(0.25))

# -- Bloc 4 : 512 filtres -> concepts haut niveau (emotion globale) --
model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))  # sortie 6x6x512
model.add(BatchNormalization())
model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))                          # -> 3x3x512
model.add(Dropout(0.25))

# -- Tete de classification --
model.add(Flatten())                                                # 3x3x512 = 4608 -> vecteur 1D

model.add(Dense(512, activation="relu"))                            # 1ere couche dense (raisonnement global)
model.add(BatchNormalization())
model.add(Dropout(0.5))                                             # dropout fort dans la tete (zone qui overfitte le plus)

model.add(Dense(256, activation="relu"))                            # 2eme couche dense (combine les features)
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(7, activation="softmax"))                           # sortie : 7 probabilites (une par emotion)

model.summary()                                                     # affiche l'architecture + nombre de parametres


# ===========================================================================
# 7. Compilation
# ===========================================================================
model.compile(
    optimizer=Adam(learning_rate=0.001),                            # Adam : adaptatif. lr=1e-3 = valeur de depart standard
    loss="sparse_categorical_crossentropy",                         # labels en entiers (pas one-hot)
    metrics=["accuracy"],                                           # metrique suivie pendant le training
)


# ===========================================================================
# 8. Callbacks (mecaniques de controle pendant le training)
# ===========================================================================
# EarlyStopping : si val_loss n'ameliore pas pendant 'patience' epochs,
#   on stoppe et on restaure les MEILLEURS poids (pas les derniers).
# ReduceLROnPlateau : si val_loss stagne, on divise le lr -> permet
#   d'affiner la descente de gradient quand on approche du minimum.
# ===========================================================================
early_stop = EarlyStopping(
    monitor="val_loss",                                             # metrique a surveiller
    patience=15,                                                    # 15 epochs sans amelioration avant d'arreter
    restore_best_weights=True,                                      # garde les poids du meilleur epoch (pas les derniers)
    verbose=1,                                                      # affiche un message quand ca declenche
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,                                                     # divise le learning rate par 2
    patience=3,                                                     # apres 3 epochs sans amelioration
    min_lr=1e-6,                                                    # plancher : ne descend pas en dessous
    verbose=1,
)

# TensorBoard : trace loss/accuracy/lr a chaque epoch + histogrammes des poids.
# Lancer ensuite : tensorboard --logdir results/logs   (sujet l. 56 : obligatoire)
import os, datetime
LOG_DIR = os.path.join("results", "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_cb = TensorBoard(
    log_dir=LOG_DIR,
    histogram_freq=1,                                               # histogrammes des poids 1x par epoch
    write_graph=True,
    update_freq="epoch",
)


# ===========================================================================
# 9. Entrainement
# ===========================================================================
print("==> Training started...")

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),                  # generateur d'images augmentees, batchs de 64
    validation_data=(X_test, y_test),                               # evalue sur le test set a chaque epoch
    epochs=80,                                                      # max 80 epochs (early stopping coupera avant si OK)
    callbacks=[early_stop, reduce_lr, tensorboard_cb],              # active les 3 callbacks (incl. TensorBoard)
    class_weight=class_weights,                                     # poids par classe pour compenser le desequilibre
    verbose=1,
)


# ===========================================================================
# 10. Evaluation finale
# ===========================================================================
loss, acc = model.evaluate(X_test, y_test, verbose=1)               # mesure finale sur le test set
print(f"ðŸŽ¯ Final Accuracy: {acc:.4f}")                              # 4 decimales pour comparer precisement


# ===========================================================================
# 11. Sauvegarde du modele
# ===========================================================================
os.makedirs(os.path.join("results", "model"), exist_ok=True)
MODEL_PATH = os.path.join("results", "model", "final_emotion_model.keras")
model.save(MODEL_PATH)                                              # sauvegarde dans results/model/ (sujet l. 84)
print(f"ðŸ’¾ Model saved to {MODEL_PATH}")


# ===========================================================================
# 12. Sauvegarde architecture (sujet l. 58 : final_emotion_model_arch.txt)
# ===========================================================================
ARCH_PATH = os.path.join("results", "model", "final_emotion_model_arch.txt")
with open(ARCH_PATH, "w") as f:
    f.write("=== Architecture finale (model.summary) ===\n\n")
    model.summary(print_fn=lambda line: f.write(line + "\n"))
    f.write(f"\n=== Resultats ===\n")
    f.write(f"Final test loss     : {loss:.4f}\n")
    f.write(f"Final test accuracy : {acc:.4f}\n")
print(f"ðŸ“ Architecture saved to {ARCH_PATH}")


# ===========================================================================
# 13. Learning curves (sujet l. 58 : learning_curves.png)
# ===========================================================================
CURVES_PATH = os.path.join("results", "model", "learning_curves.png")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history["loss"], label="train")
axes[0].plot(history.history["val_loss"], label="val")
axes[0].set_title("Loss")
axes[0].set_xlabel("epoch")
axes[0].legend()

axes[1].plot(history.history["accuracy"], label="train")
axes[1].plot(history.history["val_accuracy"], label="val")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("epoch")
axes[1].legend()

plt.tight_layout()
plt.savefig(CURVES_PATH, dpi=120)
plt.close(fig)
print(f"ðŸ“ˆ Learning curves saved to {CURVES_PATH}")