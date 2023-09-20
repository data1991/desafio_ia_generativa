# <img src="https://avatars1.githubusercontent.com/u/26231823?s=280&v=4" width="80" height="80"> Desafio IA Generativa com Python
Desafio de Projeto - Explorando IA Generativa em um Pipeline de ETL com Python


### Link do Projeto

Visite o Desafio [Projeto](https://colab.research.google.com/drive/1lqwelLZn4aeQKYh3lJUSRFuUxkLReYB6#scrollTo=jkmrk8RdooP9)

### Ferramentas utilizadas

<img src="https://www.bing.com/images/search?view=detailV2&ccid=8SHG1OHr&id=5309435A5E38677EA49ADD545B11EAEC5C42BBC3&thid=OIP.8SHG1OHrBwCvUV7i_xPGpAHaHa&mediaurl=https%3A%2F%2Fstatic.vecteezy.com%2Fsystem%2Fresources%2Fpreviews%2F022%2F227%2F364%2Fnon_2x%2Fopenai-chatgpt-logo-icon-free-png.png&cdnurl=https%3A%2F%2Fth.bing.com%2Fth%2Fid%2FR.f121c6d4e1eb0700af515ee2ff13c6a4%3Frik%3Dw7tCXOzqEVtU3Q%26pid%3DImgRaw%26r%3D0&exph=980&expw=980&q=logo+chat+gpt+png&form=IRPRST&ck=BE129CF4ED74F884A89B2748865CE579&selectedindex=1&ajaxhist=0&ajaxserp=0&vt=2" width="40" height="40"/>
<img loading="lazy" src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" width="40" height="40"/>

### Objetivo do Projeto: 
Criar um modelo GAN para gerar imagens de sequenciamento genético fictícias em preto e branco.

### Veja o passo-a-passo desse projeto e refassa ou melhore-o 

### Passo 1: Importar Bibliotecas

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten, Input
from tensorflow.keras.optimizers import Adam

### Passo 2: Geração de Dados Sintéticos

def generate_synthetic_data(num_samples, sequence_length):
    synthetic_data = []
    for _ in range(num_samples):
        sequence = ''.join(np.random.choice(['A', 'C', 'G', 'T'], sequence_length))
        synthetic_data.append(sequence)
    return synthetic_data

num_samples = 1000
sequence_length = 50
synthetic_data = generate_synthetic_data(num_samples, sequence_length)


### Passo 3: Pré-processamento de Dados (ETL)

def create_image(sequence): 

    base_colors = {'A': 0, 'C': 127, 'G': 255, 'T': 255}

    # Inicializa uma matriz vazia para a imagem
    image = np.zeros((len(sequence), 1), dtype=np.uint8)

    # Preenche a imagem com base nas sequências de DNA
    for i, base in enumerate(sequence):
        image[i, 0] = base_colors[base]
    return image
synthetic_images = [create_image(sequence) for sequence in synthetic_data]
synthetic_images = np.array(synthetic_images)
synthetic_images = synthetic_images / 255.0  


### Passo 4: Criar o Modelo Generator

def build_generator():
    model = Sequential()
    model.add(Dense(256, input_dim=100))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(synthetic_images.shape[1] * synthetic_images.shape[2], activation='tanh'))
    model.add(Reshape((synthetic_images.shape[1], synthetic_images.shape[2], 1)))
    return model

generator = build_generator()


### Passo 5: Criar o Modelo Discriminator

def build_discriminator(input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(1024))
    model.add(LeakyReLU(0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model

discriminator = build_discriminator(synthetic_images.shape[1:])


### Passo 6: Criar o Modelo GAN

def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = Input(shape=(100,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(gan_input, gan_output)
    return gan

gan = build_gan(generator, discriminator)


### Passo 7: Compilar os Modelos

discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

### Passo 8: Treinar o Modelo GAN

def train_gan(generator, discriminator, gan, synthetic_images, epochs=1, batch_size=128):
    batch_count = synthetic_images.shape[0] // batch_size

    for e in range(epochs + 1):
        for _ in range(batch_count):
            noise = np.random.normal(0, 1, size=[batch_size, 100])
            generated_images = generator.predict(noise)
            image_batch = synthetic_images[np.random.randint(0, synthetic_images.shape[0], size=batch_size)]
            discriminator.trainable = True
            d_loss_real = discriminator.train_on_batch(image_batch, np.ones((batch_size, 1)))
            d_loss_fake = discriminator.train_on_batch(generated_images, np.zeros((batch_size, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            noise = np.random.normal(0, 1, size=[batch_size, 100])
            discriminator.trainable = False
            g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

        print(f"Epoch {e}/{epochs}, D Loss: {d_loss}, G Loss: {g_loss}")

        if e % 10 == 0:
            plot_generated_images(e, generator)

def plot_generated_images(epoch, generator, examples=10, dim=(1, 10), figsize=(10, 1)):
    noise = np.random.normal(0, 1, size=[examples, 100])
    generated_images = generator.predict(noise)
    plt.figure(figsize=figsize)
    for i in range(examples):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"gan_generated_image_epoch_{epoch}.png")
    plt.show()

train_gan(generator, discriminator, gan, synthetic_images, epochs=100, batch_size=128)




