# dcgan-autoencoder

This is a Theano implementation of an convolutional autoencoder trained with an adversarial network loss function.  The code in the ipython notebook owes a lot to the implementation given by [Alec Radford et al}(https://github.com/Newmu/dcgan_code).  

## How to run

The following should work on unix systems.  Working in a virtualenv, run 

```pip install -r /path/to/requirements.txt```

You should download the CelebA dataset from [website](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) (you're looking for a file called img_align_celeba.zip).  Unzip into this directory then run 

``` ./dataprocessing.py ``` 

This will crop the images to the right size and store them in HDF5 format.

Next run the dcgan notbook.
