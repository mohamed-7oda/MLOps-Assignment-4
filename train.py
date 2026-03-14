import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import zipfile
import os

# import mlflow
import mlflow.keras

# Telling MLflow to Save everything inside ./mlruns instead of the Windows user directory
mlflow.set_tracking_uri("file:./mlruns")

# Put the experimen named
mlflow.set_experiment("Assignment3_AhmedElrashidy")

# =========================
# Unzip dataset
# =========================

zip_path = "mnist_csv.zip"

if not os.path.exists("mnist_data"):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("mnist_data")
    print("Unzipped successfully!")

# =========================
# Load data
# =========================

data = pd.read_csv("mnist_data/mnist_train.csv")
print("Dataset shape:", data.shape)

# Preprocessing
images = data.iloc[:, 1:].values
images = (images - 127.5) / 127.5
images = images.reshape(-1, 28, 28, 1)

# =========================
# Build Generator
# =========================

def build_generator():
    model = models.Sequential([
        layers.Dense(256, input_dim=100),
        layers.LeakyReLU(0.2),
        layers.Dense(512),
        layers.LeakyReLU(0.2),
        layers.Dense(1024),
        layers.LeakyReLU(0.2),
        layers.Dense(28*28*1, activation='tanh'),
        layers.Reshape((28, 28, 1))
    ])
    return model

# =========================
# Build Discriminator
# =========================

def build_discriminator():
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28, 1)),
        layers.Dense(512),
        layers.LeakyReLU(0.2),
        layers.Dense(256),
        layers.LeakyReLU(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

generator = build_generator()
discriminator = build_discriminator()

discriminator.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# =========================
# Build GAN
# =========================

discriminator.trainable = False

gan_input = layers.Input(shape=(100,))
generated_image = generator(gan_input)
gan_output = discriminator(generated_image)

gan = models.Model(gan_input, gan_output)
gan.compile(optimizer='adam', loss='binary_crossentropy')

# =========================
# Training
# =========================

learning_rate = 0.1
epochs = 1000
batch_size = 128
half_batch = batch_size // 2

# Put the MLflow run
with mlflow.start_run():

    # log parameters
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)

    # Set the tag
    mlflow.set_tag("student_id", "202202236")

    # Training loop
    for epoch in range(epochs):

        idx = np.random.randint(0, images.shape[0], half_batch)
        real_images = images[idx]

        noise = np.random.normal(0, 1, (half_batch, 100))
        fake_images = generator.predict(noise, verbose=0)

        d_loss_real = discriminator.train_on_batch(real_images, np.ones((half_batch,1)))
        d_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((half_batch,1)))

        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (batch_size, 100))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size,1)))

        if epoch % 100 == 0:

            # Save Metrices
            mlflow.log_metric("discriminator_loss", d_loss[0], step=epoch)
            mlflow.log_metric("discriminator_accuracy", d_loss[1], step=epoch)
            mlflow.log_metric("generator_loss", g_loss, step=epoch)
            print(f"Epoch {epoch} | D Loss: {d_loss[0]:.4f} | D Acc: {d_loss[1]:.4f} | G Loss: {g_loss:.4f}")

    # Save the Model
    mlflow.keras.log_model(generator, "generator_model")
print("Training finished.")


# Activate Environment: conda activate env_rl_project
# To see the UI of MLflow write this in the terminal: mlflow ui --backend-store-uri ./mlruns --port 5000
# and then visit this in the browser: http://localhost:5000
# Then make a new run if you want: python train.py