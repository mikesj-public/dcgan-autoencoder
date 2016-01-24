# dcgan-autoencoder

I recommend you look at the [write up of this repo](https://swarbrickjones.wordpress.com/2016/01/24/generative-adversarial-autoencoders-in-theano/) before proceeding.

This is a Theano implementation of an convolutional autoencoder trained with an adversarial network loss function.  This structure was used to try and upscale some grainy images of celebrities, [written up here](https://swarbrickjones.wordpress.com/2016/01/13/enhancing-images-using-deep-convolutional-generative-adversarial-networks-dcgans/).

Example output - for each triple of images, the one on the left is the original image, the middle one is the grainy version given to the autoencoder and finally the one on the right is the neural network's attempt to reconstruct the original ![My image](https://swarbrickjones.files.wordpress.com/2016/01/1452706493.png)

The code in the ipython notebook closely follows the implementation given by [Alec Radford et al](https://github.com/Newmu/dcgan_code).  

## How to run

I assume knowledge of IPython (Jupyter), pip and virtualenv (not complicated to learn if not).  The following should work on unix systems.  Working in a virtualenv, run 

```pip install -r /path/to/requirements.txt```

You should download the CelebA dataset from [website](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) (you're looking for a file called img_align_celeba.zip).  Unzip into this directory then run 

``` ./dataprocessing.py ``` 

This will crop the images to the right size and store them in HDF5 format.

Next run the dcgan notbook.
