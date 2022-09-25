# fft_for_alpa
Adds FFT functions to Alpa, for Jax.

[Alpa](https://github.com/alpa-projects/alpa) is a great auto-parallelization package that automates large-scale distribution of Jax models. [Unfortunately, it is incompatible with Jax's implementations of FFTs (as of Alpa version 0.2.0)](https://github.com/alpa-projects/alpa/issues/713). This repo provides easy-to-use FFT, IFFT, RFFT, and IRFFT functions that are compatible with Alpa.

You may want to use this code if you want to do an FFT convolution. Use cases include applying a convolution to a flattened image or to an embedding. [This text provides more uses](https://www.analog.com/media/en/technical-documentation/dsp-book/dsp_book_ch18.pdf)

In some cases, cuDNN is supposed to automatically detect certain criteria and automatically convert convolutions on long sequences to FFT convolutions, but I have found that this process is not always succesful and so the FFT transformation must be done manually. This failure can occur from virtually indefinite compiling that can take many hours. This can compile within minutes.

## Issues

**Currently, fft_for_alpa supports evenly-shaped, 1D data only**

**Sequences of length >=16,384 may hang indefinitely on compile**

**While the average FFT/RFFT and IFFT/IRFFT error is less than 2% in the included testing, a rare divergence with Jax's FFT function can result with very long sequences where a single value in the array may be off by ~25%+.**

## Usage
Put fft_for_alpa.py in your source folder and import it as shown below.

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
        out = apply_fn(model_state, params, batch["x"], batch["x"].shape) # Add the shape as an argument to the model
        return jnp.mean((out - batch["y"]) ** 2)

    grads = grad(loss_func)(model_state.params)
    new_model_state = model_state.apply_gradient(grads)
    return new_model_state

# The training loop now automatically runs on your designated cluster
model_state = create_train_state()
for batch in data_loader:
    model_state = train_step(model_state, batch)
```


### NLP Example
```python


import alpa
import fft_for_alpa

def apply_fn(model_state, params, x, x_shape):
    (fft, ifft, rfft, irfft,) = fft_for_alpa.get_fft_functions(x_shape[-1])  
    k = self.params("convolution_kernel")
    
    xd = rfft(x)
    kd = rfft(k)
    
    out = irfft(xd * kd)
    return out

# Parallelize the training step in Jax by simply using a decorator
@alpa.parallelize
def train_step(model_state, batch):
    def loss_func(params):
        out = apply_fn(model_state, params, batch["x"], batch["x"].shape) # Add the shape as an argument to the model
        return jnp.mean((out - batch["y"]) ** 2)

    grads = grad(loss_func)(model_state.params)
    new_model_state = model_state.apply_gradient(grads)
    return new_model_state

# The training loop now automatically runs on your designated cluster
model_state = create_train_state()
for batch in data_loader:
    model_state = train_step(model_state, batch)
```
