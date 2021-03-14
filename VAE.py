import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
# import tensorflow_probability as tfp
import time
import os
import argparse
from tensorflow.keras.layers import Input, Dropout, Dense, InputLayer, Conv2D, Flatten, Dense, Lambda, Reshape, Conv2DTranspose
from tensorflow.keras.models import  Model

class VAE():
    def __init__(self,dataset_name):

        
        X_train, y_train = self.load_data(dataset_name)
        self.img_rows = X_train.shape[1]
        self.img_cols = X_train.shape[2]
        self.img_channels = X_train.shape[3]
        self.img_size = X_train.shape[1] * X_train.shape[2] * X_train.shape[3]
        self.img_shape = (self.img_rows, self.img_cols, self.img_channels)
        self.z_dim = 2
        self.dataset_name = dataset_name
        # self.X_train = X_train
        # self.y_train = y_train

        # Build and compile the discriminator
        self.vae = self.build_vae()
        self.encoder.summary()
        self.decoder.summary()
    def build_vae(self):

        optimizer =  tf.keras.optimizers.Adam(1e-3) #
        n_pixels = self.img_rows*self.img_cols*self.img_channels


        #encoder
        input_img = Input(shape=self.img_shape)
        x = Conv2D(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')(input_img)
        x = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation='relu')(x)
        x = Flatten()(x)
        x = Dense(16, activation='relu')(x)
        


        # mean and variance parameters
        z_mean = Dense(self.z_dim)(x)
        z_log_var = Dense(self.z_dim)(x)

        #sample the latent vector
        z_rand = Lambda(self.sampling, output_shape=(self.z_dim,))([z_mean, z_log_var])
        #save the encoder
        self.encoder = Model(input_img, [z_mean, z_log_var, z_rand], name='encoder')
        

        #build decoder
        latent_inputs = Input(shape=(self.z_dim,), name='z_sampling')
        y = Dense(units= 7*7*64, activation='relu')(latent_inputs)
        y = Reshape(target_shape=(7, 7, 64))(y)
        y = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same',activation='relu')(y)
        y = Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same',activation='relu')(y)
        decoder_outputs = Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same')(y)

        self.decoder = Model(latent_inputs, decoder_outputs, name='decoder')

        #build encoder + decoder (total model)
        output_img = self.decoder(self.encoder(input_img)[2])
        vae = Model(input_img, output_img, name='vae_cnn')


        #define the loss
        vae_loss = self.vae_loss(input_img, output_img, z_mean, z_log_var)
        vae.add_loss(vae_loss)
        vae.compile(optimizer=optimizer)

        return vae

    def vae_loss(self, data, reconstruction, z_mean, z_log_var):
        # BEGIN INSERT CODE

        # reconstruction loss
        
        reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(tf.keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2))
            )
        # kl loss
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        # total loss
        total_loss = reconstruction_loss + kl_loss


        return total_loss

    def sampling(self,args):
        #Reparameterization trick
        z_mean, z_log_var = args
        batch_size = tf.shape(z_mean)[0]
        epsilon = tf.keras.backend.random_normal(shape=(batch_size, self.z_dim)) # create random normal vector
        z_rand = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        return z_rand

    def preprocess_images(self,images):
        images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
        #images = np.where(images > .5, 1.0, 0.0).astype('float32')
        return images

    def load_data(self,dataset_name):
        # Load the dataset
        if(dataset_name == 'mnist'):
            (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        elif(dataset_name=='fashion_mnist'):
            (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

        else:
            print('Error, unknown database')

        # normalise images between 0 and 1
        #X_train = X_train/255.0
        X_train = self.preprocess_images(X_train)
        #add a channel dimension, if need be (for mnist data)
        if(X_train.ndim ==3):
            X_train = np.expand_dims(X_train, axis=3)
        return X_train, y_train
    

    def train(self, n_iters=5000, batch_size=128, sample_interval=50):
		
		#load dataset
        X_train,_ = self.load_data(self.dataset_name)
        
        for i in range(0, n_iters):

            # ---------------------
            #  Train variational autoencoder
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            curr_batch = X_train[idx, :, :, :]
            # Autoencoder training
            loss = self.vae.train_on_batch(curr_batch, None)

            # print the losses
            print("%d [Loss: %f]" % (i, loss))

            # Save some random generated images and the models at every sample_interval iterations
            if (i % sample_interval == 0):
                n_images = 5
                idx = np.random.randint(0, X_train.shape[0], n_images)
                test_imgs = X_train[idx, :, :, :]
                self.reconstruct_images(test_imgs,'images_vae/'+self.dataset_name+'_reconstruction_%06d.png' % i)
                self.sample_images('images_vae/'+self.dataset_name+'_random_samples_%06d.png' % i)

    def reconstruct_images(self, test_imgs, image_filename):
        n_images = test_imgs.shape[0]
        #get output images
        output_imgs = np.reshape(self.vae.predict( test_imgs ),(n_images,self.img_rows,self.img_cols,self.img_channels))
        r = 2
        c = n_images
        fig, axs = plt.subplots(r, c)
        for j in range(c):
            #black and white images
            axs[0,j].imshow(test_imgs[j, :, :, 0], cmap='gray')
            axs[0,j].axis('off')
            axs[1,j].imshow(output_imgs[j, :, :, 0], cmap='gray')
            axs[1,j].axis('off')
        fig.savefig(image_filename)
        plt.close()

    def sample_images(self, image_filename):

        n_images = 8	#number of random images to sample
        #get output images
        z_sample = np.random.normal(0,1,(n_images,self.z_dim))
        r = 1
        c = n_images
        fig, axs = plt.subplots(r, c)
        for j in range(c):
            x_decoded = np.reshape(self.decoder.predict(z_sample) , (n_images,self.img_rows,self.img_cols,self.img_channels))
            #black and white images
            axs[j].imshow(x_decoded[j, :, :, 0], cmap='gray')
            axs[j].axis('off')
        fig.savefig(image_filename)
        plt.close()
    def plot_label_clusters(self, data, labels):
        if self.z_dim == 2:
            # display a 2D plot of the digit classes in the latent space
            z_mean, _, _ = self.encoder.predict(data)
            plt.figure(figsize=(12, 10))
            plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
            plt.colorbar()
            plt.xlabel("z[0]")
            plt.ylabel("z[1]")
            plt.savefig('images_vae/label_cluster.png')
            # plt.show()
            plt.close()
        else:
            print(' It is not implemented for zdim >2 !')


    def plot_latent_space(self, n=30, figsize=15):
        if self.z_dim == 2:
            # display a n*n 2D manifold of digits
            digit_size = 28
            scale = 1.0
            figure = np.zeros((digit_size * n, digit_size * n))
            # linearly spaced coordinates corresponding to the 2D plot
            # of digit classes in the latent space
            grid_x = np.linspace(-scale, scale, n)
            grid_y = np.linspace(-scale, scale, n)[::-1]

            for i, yi in enumerate(grid_y):
                for j, xi in enumerate(grid_x):
                    z_sample = np.array([[xi, yi]])
                    x_decoded = self.decoder.predict(z_sample)
                    digit = x_decoded[0].reshape(digit_size, digit_size)
                    figure[
                        i * digit_size : (i + 1) * digit_size,
                        j * digit_size : (j + 1) * digit_size,
                    ] = digit

            plt.figure(figsize=(figsize, figsize))
            start_range = digit_size // 2
            end_range = n * digit_size + start_range
            pixel_range = np.arange(start_range, end_range, digit_size)
            sample_range_x = np.round(grid_x, 1)
            sample_range_y = np.round(grid_y, 1)
            plt.xticks(pixel_range, sample_range_x)
            plt.yticks(pixel_range, sample_range_y)
            plt.xlabel("z[0]")
            plt.ylabel("z[1]")
            plt.imshow(figure, cmap="Greys_r")
            plt.savefig('images_vae/latent_space.png')
            # plt.show()
            plt.close()
        else:
            print(' It is not implemented for zdim >2 !')



if __name__ == '__main__':
    if (os.path.isdir('images_vae')==0):
        os.mkdir('images_vae')

    parser = argparse.ArgumentParser(prog='PROG')
    parser.add_argument('-d', type=str, default='mnist', help='dataset_name')
    parser.add_argument('-n', type=int, default=10000, help='number of iteration')
    parser.add_argument('-b', type=int, default=128, help=' batch_size')
    parser.add_argument('-s', type=int, default=1000, help='sample_interval')
    args = parser.parse_args()

	
	#create AE model
	
    vae = VAE(args.d)
    vae.train(n_iters=args.n, batch_size=args.b, sample_interval=args.s)
    if vae.z_dim == 2:
        # PLot latent space
        data,labels = vae.load_data(vae.dataset_name)
        vae.plot_label_clusters(data,labels)
        # PLot Mnist manifold
        vae.plot_latent_space()