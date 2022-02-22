from tensorflow import keras
from keras.layers import Input, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from functools import partial
import keras.backend as K

from tqdm import tqdm
import matplotlib.pyplot as plt
import time

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


class WGANGP():
    def __init__(self, train_dataset, optimizer_status = 1, no_neurons = 256, no_layers = 1, repetition= 1):
        self.train_dataset = train_dataset
        self.no_neurons = no_neurons
        self.no_layers = no_layers
        self.folder_address = f'models/adam_ornot{optimizer_status}trained_models_{no_neurons}neurons_{no_layers}layers{repetition}'
        self.batch_size = 140 
        self.serie_dim = 24
        self.noise_dim = 100
        self.con_dim = 28
        self.disc_input_dim = self.serie_dim+self.con_dim
        self.gen_input_dim = self.noise_dim+self.con_dim
        self.d_loss = []
        self.g_loss = []
        self.noise_con_series = []
        self.real_price_con_series = []

        self.n_discriminator = 5
        self.n_generator = 1
        if optimizer_status:
            disc_optimizer = keras.optimizers.Adam(learning_rate=0.001)
            gen_optimizer = keras.optimizers.Adam(learning_rate=0.001)
        else:
            disc_optimizer = keras.optimizers.RMSprop()
            gen_optimizer = keras.optimizers.RMSprop()
        # Build the generator and discriminator
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        #-------------------------------
        # Construct Computational Graph
        #       for the discriminator
        #-------------------------------

        # Freeze generator's layers while training discriminator
        self.generator.trainable = False

        # Image input (real sample)
        real_price_con_serie = Input(shape=self.disc_input_dim)

        # Noise input
        noise_con_serie = Input(shape=self.gen_input_dim)
        # Generate image based of noise (fake sample)
        con_serie = real_price_con_serie[:,-1*self.con_dim::]
        fake_price_serie = self.generator(noise_con_serie)
        fake_price_con_serie = tf.concat((fake_price_serie,con_serie),axis=1)

        # Discriminator determines validity of the real and fake images
        fake_validity = self.discriminator(fake_price_con_serie)
        real_validity = self.discriminator(real_price_con_serie)

        # Construct weighted average between real and fake images
        interpolated_sample = self.interpolate_sample(real_price_con_serie,fake_price_con_serie)
        # Determine validity of weighted sample
        validity_interpolated = self.discriminator(interpolated_sample)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolated_sample)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names
        self.discriminator_model = Model(inputs=[real_price_con_serie, noise_con_serie],
                            outputs=[real_validity, fake_validity, validity_interpolated])
        self.discriminator_model.compile(loss=[self.wasserstein_loss,
                                              self.wasserstein_loss,
                                              partial_gp_loss],
                                        optimizer=disc_optimizer,
                                        loss_weights=[1, 1, 10], experimental_run_tf_function=False)
        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze the discriminator's layers
        self.discriminator.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        noise_con_serie = Input(shape=(self.gen_input_dim,))
        # Generate images based of noise
        fake_price_serie = self.generator(noise_con_serie)
        fake_price_con_serie = tf.concat((fake_price_serie,noise_con_serie[:,-1*self.con_dim::]),axis=1)
        # Discriminator determines validity
        fake_validity = self.discriminator(fake_price_con_serie)
        # Defines generator model
        self.generator_model = Model(noise_con_serie, fake_validity)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=gen_optimizer,experimental_run_tf_function=False)

    def interpolate_sample(self, real_samples, fake_samples):
        alpha = tf.random.uniform((self.batch_size,1), 0.0, 1.0)
        diff = fake_samples - real_samples
        interpolated = real_samples + alpha * diff
        return interpolated

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """

        gradients = K.gradients((y_pred+1e-5), (averaged_samples))[0]
        # compute the euclidean norm by squaring
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)

    def wasserstein_loss(self, y_true, y_pred):
        loss = K.mean((y_true+1e-5) * (y_pred+1e-5))
        return loss

    def build_generator(self):
        model = Sequential()
        model.add(Dense(self.no_neurons, input_dim=self.gen_input_dim))
        model.add(LeakyReLU(alpha=0.2))
        if self.no_layers > 1:
            for _ in range(self.no_layers-1):
                model.add(Dense(self.no_neurons))
                model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(self.serie_dim, activation='linear'))

        model.summary()

        input_noise_con_serie = Input(shape=self.gen_input_dim)
        g_price_serie = model(input_noise_con_serie)
        return Model(input_noise_con_serie, g_price_serie)

    def build_discriminator(self):
        model = Sequential()
        model.add(Dense(256, input_dim=self.disc_input_dim))
        model.add(LeakyReLU(alpha=0.2))
        if self.no_layers > 1:
            for _ in range(self.no_layers-1):
                model.add(Dense(self.no_neurons))
                model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        input_price_con_serie = Input(shape=self.disc_input_dim)
        validity = model(input_price_con_serie)
        return Model(input_price_con_serie, validity)

    def plot_save_real_fake_samples(self,epoch):
        c = 7

        real_sample_con = self.train_dataset[[100,101,102,103,104,105,106],:]
        con_serie = real_sample_con[:,-1*self.con_dim::]

        r_decision = self.discriminator.predict(real_sample_con)
        
        random_latent_vectors = np.random.normal(0,1,(c, self.noise_dim))
        noise_con_serie = np.hstack((random_latent_vectors, con_serie))

        generated_sample = self.generator.predict(noise_con_serie)
        generated_sample_con = np.hstack((generated_sample, con_serie))

        g_decision = self.discriminator.predict(generated_sample_con)

        fig, axs = plt.subplots(ncols=c, figsize=(35,5))
        cnt = 0
        for j in range(c):
            axs[j].plot(real_sample_con[cnt,0:self.serie_dim],'r',label='real')
            axs[j].plot(generated_sample_con[cnt,0:self.serie_dim],'b',label='fake')
            axs[j].set_title(f"$F_M$ & $F_D$: {con_serie[cnt,[0,1]]}" )
            axs[j].set_xlabel(f'Validity: real{r_decision[cnt]}, fake{g_decision[cnt]}')
            axs[j].legend()
            axs[j].set_ylim([-1.5,1.5])
            cnt += 1
        plt.savefig(self.folder_address+f'/real_fake_samples_at_epoch_{epoch}.pdf')
        plt.close()
        
    def train(self,epoch_start, epoch_end, sample_interval, train_dataset, training_time = 60*60):
        start_time = time.time()
        batch_size = self.batch_size
        # Load the dataset
        X_train = train_dataset

        # Adversarial ground truths
        valid = -tf.ones((batch_size, 1))
        fake =  tf.ones((batch_size, 1))
        dummy = tf.zeros((batch_size, 1)) # Dummy gt for gradient penalty
        for epoch in tqdm(range(epoch_start, epoch_end)):
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_price_con_serie = X_train[idx]
                # Sample generator input
            noise = tf.random.normal((batch_size, self.noise_dim), 0, 1)
            noise_con_serie = tf.concat((noise,real_price_con_serie[:,-1*self.con_dim::]),axis=1)
            for _ in range(self.n_discriminator):

                # ---------------------
                #  Train Discriminator
                # ---------------------
                d_loss = self.discriminator_model.train_on_batch([real_price_con_serie, noise_con_serie],
                                                                [valid, fake, dummy])
                self.d_loss.append(d_loss[0])

            # ---------------------
            #  Train Generator
            # ---------------------
            for _ in range(self.n_generator):
                g_loss = self.generator_model.train_on_batch(noise_con_serie, valid)
                self.g_loss.append(g_loss)
            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))

            # If at save interval, save the models and generated image samples
            if epoch % sample_interval == 0:
                self.generator.save(self.folder_address+f'/gen_at_epoch_{epoch}.h5')
                self.discriminator.save(self.folder_address+f'/disc_at_epoch_{epoch}.h5')
                self.plot_save_real_fake_samples(epoch)
                # Also detect whether the generator is winning the adversarial learning
                if self.g_loss[-1] < -0.95:
                    return print('Early stopping because the generator is prevailing.')
                elapsed_time = time.time() - start_time
                if elapsed_time > training_time:
                    return print('Early stopping due to training time limit.')


# Specify neural network dimensions. Add in the list if you want to train several networks with different dimension.
no_neurons_list = [128]
no_layers_list = [2]
adam_or_not = [0]

input_serie_unscaled = np.load('input_serie.npy')
scaler = StandardScaler()
input_serie = scaler.fit_transform(input_serie_unscaled)
input_serie_train, input_serie_test = train_test_split(input_serie, test_size=0.2,shuffle=True,random_state=0)

# Train every neural network with specified dimension.
for optimizer_status in adam_or_not:
    for no_neurons in no_neurons_list:
        for no_layers in no_layers_list:
            wgan = WGANGP(input_serie_train,optimizer_status,no_neurons,no_layers,1)
            epoch_start = 0
            epoch_end = 30001
            wgan.train(epoch_start, epoch_end, sample_interval=200, train_dataset=input_serie)
            np.save(wgan.folder_address+'/g_loss',wgan.g_loss)
            np.save(wgan.folder_address+'/d_loss',wgan.d_loss)
            
            K.clear_session()






