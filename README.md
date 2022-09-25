# fft_for_alpa
Adds FFT functions to Alpa, for Jax.

[Alpa](https://github.com/alpa-projects/alpa) is a great auto-parallelization package that automates large-scale distribution of Jax models. [Unfortunately, it is incompatible with Jax's implementations of FFTs (as of Alpa version 0.2.0)](https://github.com/alpa-projects/alpa/issues/713). This repo provides easy-to-use FFT, IFFT, RFFT, and IRFFT functions that are compatible with Alpa.

You may want to use this code if you want to do an FFT convolution. Use cases include applying a convolution to a flattened image or to an embedding. [This text provides more uses.](https://www.analog.com/media/en/technical-documentation/dsp-book/dsp_book_ch18.pdf)

In some cases, cuDNN is supposed to automatically detect certain criteria and automatically convert convolutions on long sequences to FFT convolutions, but I have found that this process is not always succesful and so the FFT transformation must be done manually. This failure can occur from virtually indefinite compiling that can take many hours. This can compile within minutes.

## Issues

Currently, fft_for_alpa supports evenly-shaped, 1D data only

Sequences of length >=16,384 may hang indefinitely on compile

While the average FFT/RFFT and IFFT/IRFFT error is less than 2% in the included testing, a rare divergence with Jax's FFT functions may result with very long sequences where at least one value in the returned array may be off by ~25%+. For what it's worth, my models are training fine using these rfft/irfft functions.

Unit testing must be made more thorough:
["To verify the correctness of an FFT implementation, rigorous guarantees can be obtained in O(N log N) time by a simple procedure checking the linearity, impulse-response, and time-shift properties of the transform on random inputs (Ergün, 1995)."](https://en.wikipedia.org/wiki/Fast_Fourier_transform#Computational_issues:~:text=To%20verify%20the%20correctness%20of%20an%20FFT%20implementation%2C%20rigorous%20guarantees%20can%20be%20obtained%20in%20O(N%C2%A0log%C2%A0N)%20time%20by%20a%20simple%20procedure%20checking%20the%20linearity%2C%20impulse%2Dresponse%2C%20and%20time%2Dshift%20properties%20of%20the%20transform%20on%20random%20inputs%20(Erg%C3%BCn%2C%201995).%5B39%5D)



## Installation
Put fft_for_alpa.py in your project folder and import it as shown below:

## Usage
You only have to add the data shape to your apply_fn inputs and pass it to get_fft_functions() **inside** of an alpa.parallelize decoration. 


### Basic usage case
```python

import alpa
import fft_for_alpa

def apply_fn(model_state, params, x, x_shape):
    (fft, ifft, rfft, irfft,) = fft_for_alpa.get_fft_functions(x_shape[-1])
    x = rfft(x)
    x = irfft(x)
    return x    

# Parallelize the training step in Jax by simply using a decorator
@alpa.parallelize
def train_step(model_state, batch):
    def loss_func(params):
        # Add the shape as an argument to the model
        out = apply_fn(model_state, params, batch["x"], batch["x"].shape) 
        return jnp.mean((out - batch["y"]) ** 2)

    grads = grad(loss_func)(model_state.params)
    new_model_state = model_state.apply_gradient(grads)
    return new_model_state

# The training loop now automatically runs on your designated cluster
model_state = create_train_state()
for batch in data_loader:
    model_state = train_step(model_state, batch)
```


### FFT Convolution Example
```python


import alpa
import fft_for_alpa

def apply_fn(model_state, params, x, x_shape):
    ## rfft is faster
    (fft, ifft, rfft, irfft,) = fft_for_alpa.get_fft_functions(x_shape[-1])  
    k = self.params("convolution_kernel")
    
    ## Calculate the transformed versions of x and k
    xd = rfft(x)
    kd = rfft(k)
    
    ## Multiply the transformed versions, and then perform an inverse transform
    out = irfft(xd * kd)
    return out

# Parallelize the training step in Jax by simply using a decorator
@alpa.parallelize
def train_step(model_state, batch):
    def loss_func(params):
        # Add the shape as an argument to the model
        out = apply_fn(model_state, params, batch["x"], batch["x"].shape) 
        return jnp.mean((out - batch["y"]) ** 2)

    grads = grad(loss_func)(model_state.params)
    new_model_state = model_state.apply_gradient(grads)
    return new_model_state

# The training loop now automatically runs on your designated cluster
model_state = create_train_state()
for batch in data_loader:
    model_state = train_step(model_state, batch)
```
