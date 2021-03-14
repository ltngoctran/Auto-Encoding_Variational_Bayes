# Auto-Encoding_Variational_Bayes
1. Required packages:
- Tensorflow 2. 
- Numpy, Matplotlib
- Argparse
2. How to run 
 
   Run the command:

   python -m VAE -d 'dataset_name' -n number_iterations -b batch_size -s sample_interval

   Example: python -m VAE -n 20000 -b 64 -s 2000

   If we do not type 'dataset_name', number_iterations, batch_size, sample_interval, the program will set default value.