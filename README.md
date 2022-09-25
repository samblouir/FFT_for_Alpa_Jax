# fft_for_alpa
Adds FFT functions to Alpa, for Jax.

[Alpa](https://github.com/alpa-projects/alpa) is an auto-parallelization that automates large-scale distributed training. [Unfortunately, it is incompatible with Jax's implementations of FFTs (as of Alpa version 0.2.0)](https://github.com/alpa-projects/alpa/issues/713). This repo provides easy-to-use FFT, IFFT, RFFT, and IRFFT functions that are compatible with Alpa.

You may want to use this code if you want to do an FFT convolution. cuDNN is supposed to automatically do this, but compiling is not always succesful and it must be done manually.
This may be when you are using a convolutional filter on a very long sequence. Use cases include applying a convolution to a flattened image or to an embedding. [This text provides more uses](https://www.analog.com/media/en/technical-documentation/dsp-book/dsp_book_ch18.pdf)

## Issues

**Currently, fft_for_alpa supports evenly-shaped, 1D data only**

**Sequences of length >=16,384 may hang indefinitely on compile **

## Usage
Put fft_for_alpa.py in your source folder and import it as shown below.

You only have to add the data shape to your apply_fn inputs and pass it to get_fft_functions() **inside** of an alpa.parallelize decoration. 


```python
import alpa
import fft_for_alpa

def apply_fn(model_state, params, x, x_shape):
    (fft, ifft, rfft, irfft,) = alpa.get_fft_functions(x_shape[-1])
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
