import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers
from sklearn.model_selection import train_test_split
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Load dataset
(x_data, _), (_, _) = tf.keras.datasets.cifar10.load_data()

# Normalize
x_data = x_data.astype("float32") / 255.0

# Convert to grayscale
x_gray = tf.image.rgb_to_grayscale(x_data).numpy()

# Split into 80% train, 20% test
x_train_gray, x_test_gray, x_train, x_test = train_test_split(
    x_gray, x_data, test_size=0.2, random_state=42
)

print("Training set:", x_train.shape, x_train_gray.shape)
print("Testing set:", x_test.shape, x_test_gray.shape)


# ---------------------- Models ----------------------

def simple_cnn(input_shape=(32, 32, 1)):
    inputs = layers.Input(shape=input_shape)

    # Encoder 
    x = layers.Conv2D(32, (3,3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D((2,2))(x)

    # Bottleneck
    x = layers.Conv2D(64, (3,3), activation="relu", padding="same")(x)

    # Decoder 
    x = layers.UpSampling2D((2,2))(x)
    x = layers.Conv2D(32, (3,3), activation="relu", padding="same")(x)

    # Output: 3 channels (RGB)
    outputs = layers.Conv2D(3, (3,3), activation="sigmoid", padding="same")(x)

    return models.Model(inputs, outputs, name="SimpleCNN")

def unet(input_shape=(32, 32, 1)):
    inputs = layers.Input(input_shape)

    # Encoder 
    c1 = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(64, 3, activation="relu", padding="same")(p1)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Bottleneck 
    b = layers.Conv2D(128, 3, activation="relu", padding="same")(p2)

    # Decoder
    u1 = layers.UpSampling2D((2, 2))(b)
    u1 = layers.Concatenate()([u1, c2])
    c3 = layers.Conv2D(64, 3, activation="relu", padding="same")(u1)

    u2 = layers.UpSampling2D((2, 2))(c3)
    u2 = layers.Concatenate()([u2, c1])
    c4 = layers.Conv2D(32, 3, activation="relu", padding="same")(u2)

    # Output: 3 channels (RGB)
    outputs = layers.Conv2D(3, 1, activation="sigmoid")(c4)

    return models.Model(inputs, outputs, name="UNet")

def build_generator(input_shape=(32, 32, 1)):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    x = layers.Conv2D(32, (3,3), padding="same", activation="relu")(inputs)
    x = layers.Conv2D(64, (3,3), strides=2, padding="same", activation="relu")(x)

    # Bottleneck
    x = layers.Conv2D(128, (3,3), padding="same", activation="relu")(x)

    # Decoder
    x = layers.UpSampling2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), padding="same", activation="relu")(x)
    x = layers.Conv2D(32, (3,3), padding="same", activation="relu")(x)

    # Output RGB
    outputs = layers.Conv2D(3, (3,3), activation="sigmoid", padding="same")(x)

    return models.Model(inputs, outputs, name="Generator")


# ---------------- Discriminator ---------------- #
def build_discriminator(input_shape=(32, 32, 3)):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (3,3), strides=2, padding="same")(inputs)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(64, (3,3), strides=2, padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Flatten()(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    return models.Model(inputs, outputs, name="Discriminator")


# ---------------- Training ---------------- #
def train_gan(generator, discriminator, x_train_gray, x_train_rgb, epochs=50, batch_size=64, lambda_l1=100):
    # Optimizers
    d_opt = tf.keras.optimizers.Adam(1e-4)
    g_opt = tf.keras.optimizers.Adam(1e-4)

    # Losses
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    mae = tf.keras.losses.MeanAbsoluteError()

    # Metrics
    d_loss_metric = tf.keras.metrics.Mean()
    g_loss_metric = tf.keras.metrics.Mean()
    d_acc_metric = tf.keras.metrics.BinaryAccuracy()

    @tf.function
    def train_step(gray_imgs, real_imgs):
        # ---------------------
        # Train Discriminator
        # ---------------------
        with tf.GradientTape() as tape:
            fake_imgs = generator(gray_imgs, training=True)
            real_validity = discriminator(real_imgs, training=True)
            fake_validity = discriminator(fake_imgs, training=True)

            d_loss_real = bce(tf.ones_like(real_validity), real_validity)
            d_loss_fake = bce(tf.zeros_like(fake_validity), fake_validity)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)

        grads = tape.gradient(d_loss, discriminator.trainable_variables)
        d_opt.apply_gradients(zip(grads, discriminator.trainable_variables))

        d_acc_metric.update_state(tf.ones_like(real_validity), real_validity)
        d_acc_metric.update_state(tf.zeros_like(fake_validity), fake_validity)

        # -----------------
        # Train Generator
        # -----------------
        with tf.GradientTape() as tape:
            fake_imgs = generator(gray_imgs, training=True)
            fake_validity = discriminator(fake_imgs, training=False)  # D frozen

            # Hybrid loss: GAN + L1 reconstruction
            adv_loss = bce(tf.ones_like(fake_validity), fake_validity)
            l1_loss = mae(real_imgs, fake_imgs)
            g_loss = adv_loss + lambda_l1 * l1_loss

        grads = tape.gradient(g_loss, generator.trainable_variables)
        g_opt.apply_gradients(zip(grads, generator.trainable_variables))

        return d_loss, g_loss

    steps_per_epoch = x_train_gray.shape[0] // batch_size

    for epoch in range(epochs):
        d_loss_metric.reset_state()
        g_loss_metric.reset_state()
        d_acc_metric.reset_state()

        for step in range(steps_per_epoch):
            idx = np.random.randint(0, x_train_gray.shape[0], batch_size)
            gray_imgs = x_train_gray[idx]
            real_imgs = x_train_rgb[idx]

            d_loss, g_loss = train_step(gray_imgs, real_imgs)

            d_loss_metric.update_state(d_loss)
            g_loss_metric.update_state(g_loss)

        print(f"Epoch {epoch+1}/{epochs} "
              f"| D loss: {d_loss_metric.result():.4f}, acc: {100*d_acc_metric.result():.2f}% "
              f"| G loss: {g_loss_metric.result():.4f}")

    return generator

# ---------------------- Train and Evaluate ----------------------

# Build models
cnn_model = simple_cnn()
unet_model = unet()
generator = build_generator()
discriminator = build_discriminator()

# Train CNN & U-net
for model in [cnn_model, unet_model]:
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    print(f"\n🚀 Training {model.name}...\n")

    history = model.fit(
        x_train_gray, x_train,
        epochs=50, batch_size=128,
        validation_data=(x_test_gray, x_test),
        verbose=1
    )

    # Save model
    model.save(f"{model.name}.h5")

    # Predict on the entire test set
    preds = model.predict(x_test_gray, batch_size=64, verbose=1)

    # Compute PSNR & SSIM for all test images
    psnr_scores = [psnr(x_test[i], preds[i], data_range=1.0) for i in range(len(x_test))]
    ssim_scores = [ssim(x_test[i], preds[i], channel_axis=-1, data_range=1.0) for i in range(len(x_test))]

    print(f"\n📊 Evaluation for {model.name}:")
    print(f"Final Validation MAE: {history.history['val_mae'][-1]:.4f}")
    print(f"Average PSNR: {np.mean(psnr_scores):.4f}")
    print(f"Average SSIM: {np.mean(ssim_scores):.4f}")
    print("-" * 40)


# Train GAN 
print("\nTraining GAN...\n")
gan_model = train_gan(generator, discriminator, x_train_gray, x_train, epochs=50, batch_size=128)
gan_model.save("GAN.h5")
# Predict on full test set
gan_preds = gan_model.predict(x_test_gray, batch_size=64, verbose=1)

# Compute PSNR & SSIM for all test images
psnr_scores = [psnr(x_test[i], gan_preds[i], data_range=1.0) for i in range(len(x_test))]
ssim_scores = [ssim(x_test[i], gan_preds[i], channel_axis=-1, data_range=1.0) for i in range(len(x_test))]

print(f"\n📊 Evaluation for GAN:")
print(f"PSNR: {np.mean(psnr_scores):.4f}")
print(f"SSIM: {np.mean(ssim_scores):.4f}")



   

# ---------------------- Visualization ----------------------
# Take 3 test grayscale images
test_imgs = x_test_gray[:3]

# Predict with CNN and U-Net
cnn_preds = cnn_model.predict(test_imgs)
unet_preds = unet_model.predict(test_imgs)

# Predict with GAN (only generator is used)
gan_preds = generator.predict(test_imgs)

# Plot results
for i in range(3):
    plt.figure(figsize=(12, 4))

    # Input grayscale
    plt.subplot(1, 5, 1)
    plt.imshow(test_imgs[i].squeeze(), cmap="gray")
    plt.title("Input (Gray)")
    plt.axis("off")

    # Ground truth
    plt.subplot(1, 5, 2)
    plt.imshow(x_test[i])
    plt.title("Ground Truth")
    plt.axis("off")

    # CNN output
    plt.subplot(1, 5, 3)
    plt.imshow(cnn_preds[i])
    plt.title("CNN Output")
    plt.axis("off")

    # GAN output
    plt.subplot(1, 5, 4)
    plt.imshow(gan_preds[i])
    plt.title("GAN Output")
    plt.axis("off")

    # U-Net output
    plt.subplot(1, 5, 5)
    plt.imshow(unet_preds[i])
    plt.title("U-Net Output")
    plt.axis("off")

    plt.show()

