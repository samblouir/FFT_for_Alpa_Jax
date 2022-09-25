## TODO: Re-implement using Chex
## TODO: Fix rare issues with precision on long sequences (16k+)
## TODO: Support vmap
## TODO: Support odd length sequences

# import numpy as np
import jax.numpy as np
import functools
import jax

def get_fft_functions(shape:int) -> tuple:
	# Returns callables fns for (rfft, irfft, np.fft.rfft, np.fft.irfft) for a given shape. 
	assert(shape%2 == 0)
	fft = functools.partial(_fft, signal_length=shape, )
	ifft = functools.partial(_ifft, signal_length=shape,)
	rfft = functools.partial(_rfft, signal_length=shape,)
	irfft = functools.partial(_irfft, signal_length=shape)
	return (fft, ifft, rfft, irfft,)

def _rfft(x:np.ndarray, signal_length:int) -> np.ndarray:
	fw = create_fourier_weights(signal_length*2, rfft=True,)
	ft = (x @ fw)
	# ft[-1] = ft[-1].real + 0j ## Numpy
	ft = ft.at[-1].set(ft[-1].real + 0j) ## Jax
	return ft

def _irfft(y:np.ndarray, signal_length:int) -> np.ndarray:

	e = signal_length
	## Unused until support for odd signal lengths is added.
	# if (signal_length % 2 == 0):
	# 	e = signal_length
	# else:
	# 	e = signal_length-1

	to_conj = y[1:e][::-1]
	conj =  np.conjugate(to_conj)
	xn = np.concatenate([y,conj])
	xn = _ifft(xn, signal_length)
	xn = np.abs(xn)
	xn = np.real(xn)
	return xn

def _fft(x:np.ndarray, signal_length:int) -> np.ndarray:
	fw = create_fourier_weights(signal_length*2, rfft=False,)
	ft = (x @ fw)
	return ft

def _ifft(x:np.ndarray, signal_length:int) -> np.ndarray:
	## Modified from this fft implementation:
	#  https://sidsite.com/posts/fourier-nets/#:~:text=checking%20is%20to-,reconstruct,-the%20signal%3A
	double_signal_length = (signal_length * 2)
	interval = np.arange(double_signal_length)
	tvals = interval[..., np.newaxis]
	freqs = interval[np.newaxis, ...]
	arg_vals = ((2 * np.pi * tvals * freqs) / double_signal_length)
	sinusoids = ((x.real * np.cos(arg_vals) - x.imag * np.sin(arg_vals)) / double_signal_length)
	reconstructed_signal = np.sum(sinusoids, axis=1)
	reconstructed_signal = reconstructed_signal[..., :signal_length] + 0j*reconstructed_signal[..., signal_length:]
	reconstructed_signal = np.pad(reconstructed_signal, (0, signal_length))
	return reconstructed_signal


def create_fourier_weights(signal_length:int, rfft:bool=True,):  
	## Modified from this fft implementation: 
	# https://sidsite.com/posts/fourier-nets/#:~:text=the%20fast%20Fourier-,transform,-.
	if rfft:
		r_length = (signal_length // 2 + 1)
		## Unused until support for odd signal lengths is added.
		# r_length = (signal_length // 2 + 1) if (signal_length % 2 == 0) else ((signal_length + 1) // 2) 
	else:
		r_length = signal_length

	(k_vals, n_vals) = np.mgrid[0:signal_length, 0:r_length]
	theta_vals = (2 * np.pi * k_vals * n_vals / signal_length)
	theta_vals = theta_vals[..., :r_length] 
	fw = (np.cos(theta_vals) + 1j * np.sin(theta_vals))
	return (1/fw)



def __test_fn__(x:np.ndarray, shape:tuple, rtol:float = 1e-1, atol:float = 1e-1, use_allclose_tests:bool = False,):
	## allclose, rtol and atol disabled. Errors noted on long sequences (~16k length+)
	## Average case eems to be fine.

	(fft, ifft, rfft, irfft,) = get_fft_functions(shape[-1])

	x = np.pad(x, (0, shape[-1]))
	original_x = x

	def get_stats(transformed, comparison):
		s = np.argmax(np.abs(transformed - comparison))
		worst_tranformation_percentage_error = np.abs(1 - (transformed[s] / comparison[s]))
		return {
			"worst_tranformation_percentage_error":worst_tranformation_percentage_error,
			"worst_transforation_absolute_error": np.abs(transformed[s] - comparison[s]),
			# "current_signal_length": transformed.shape[-1],
			# "worst_tranformation": transformed[s],
			# "worst_tranformation_comparison": comparison[s],
		}


	def test_1(x:np.ndarray, use_allclose_tests:bool = False) -> dict:
		###################
		## Test #1
		###################
		transformed_x = rfft(x)
		inverse_transformed_x = irfft(transformed_x)

		comparison_transformed_x = np.fft.rfft(x)
		comparison_inverse_transformed_x = np.fft.irfft(comparison_transformed_x)

		if use_allclose_tests:
			assert(np.allclose(x, original_x, atol=atol, rtol=rtol,))
			assert(np.allclose(transformed_x, comparison_transformed_x, atol=atol, rtol=rtol,))
			assert(np.allclose(inverse_transformed_x, comparison_inverse_transformed_x, atol=atol, rtol=rtol,)) ## Looser tolerance for Jax
			assert(np.allclose(x, comparison_inverse_transformed_x, atol=atol, rtol=rtol,))

		return {
			"worst_rfft_error_percentage": get_stats(transformed_x, comparison_transformed_x), 
			"worst_irfft_error_percentage":  get_stats(inverse_transformed_x, comparison_inverse_transformed_x),
		}

	def test_2(x:np.ndarray, use_allclose_tests:bool = False) -> dict:
		###################
		## Test #2
		###################
		transformed_x = fft(x)
		inverse_transformed_x = ifft(transformed_x)

		comparison_transformed_x = np.fft.fft(x)
		comparison_inverse_transformed_x = np.fft.ifft(comparison_transformed_x)

		if use_allclose_tests:
			assert(np.allclose(x, original_x, atol=atol, rtol=rtol,))
			assert(np.allclose(transformed_x, comparison_transformed_x, atol=atol, rtol=rtol,))
			assert(np.allclose(inverse_transformed_x, comparison_inverse_transformed_x, atol=atol, rtol=rtol,)) ## Looser tolerance for Jax
			assert(np.allclose(x, comparison_inverse_transformed_x, atol=atol, rtol=rtol,))

		return {
			"worst_fft_error_percentage": get_stats(transformed_x, comparison_transformed_x), 
			"worst_ifft_error_percentage":  get_stats(inverse_transformed_x, comparison_inverse_transformed_x),
		}

	rfft_results = test_1(x = np.copy(original_x), use_allclose_tests=use_allclose_tests,)
	fft_results = test_2(x = np.copy(original_x), use_allclose_tests=use_allclose_tests,)

	result_dict = {
		**{f"rfft": v for k, v in rfft_results.items()},
		**{f"fft": v for k, v in fft_results.items()},
		"stats": {"x.shape":x.shape,},
	}

	## TODO: Something isn't freeing RAM between tests. OOM with 96GB of RAM noticed when doing 200+ tests.
	del x, original_x

	return result_dict
		

## Run the tests and collect results
def __run_args__(current_uniform_rngs, shape, dim):
	## Barely reminescent of junit test setup. Create a fresh "x" each time.
	## Create a new x with random shape and values
	x = jax.random.uniform(current_uniform_rngs, shape=dim[:shape],)
	try:
		result = __test_fn__(x, shape=x.shape,)  ## Raises exception o
		return result
	except Exception as e:
		print(f"  Test failure using x.shape: {x.shape}. Error: {e}")
		raise e

def __mp_args__(args):
	## For multiprocessing and TQDM, since TQDM doesn't have a starmap equivalent
	return __run_args__(*args)



def __run_tests__(fuzzes:int = 50, use_multiprocessing:bool = False, min_length:int = 1, max_length:int = 8, max_dims = 1,):
	if use_multiprocessing:
		import multiprocessing as mp
		jax.config.update('jax_platforms', 'cpu')

	print(
		f"  Running {fuzzes} tests. This may take a while...",
		f"\n  use_multiprocessing: {use_multiprocessing}",
		f", max_length: {max_length}",
		f", max_dims: {max_dims}",
		sep='',
		)

	
	## Basic tests to ensure that the fft functions are working as expected.
	## Not exhaustive, but should be sufficient for the purposes of this demo.
	## We can see some differences in the output of the fft functions vs Jax's fft functions,
	## but they are close enough with a slightly increased tolerance.
	_rng = jax.random.PRNGKey(0)
	(_rng, shape_rng, dims_rng, *uniform_rngs) = jax.random.split(_rng, fuzzes+3)

	## Better options for later testing with >1D support
	shapes = jax.random.randint(shape_rng, (fuzzes,), 1, max_dims+1)
	dims = jax.random.randint(dims_rng, (fuzzes, max_dims,), min_length, max_length)
	dims += (dims % 2) ## Makes the dims even
	## Sorts dims so we can get a progress bar up quicker by letting the smallest dims go first.
	## This won't affect the final results.
	dims = np.sort(dims)
		
	## Prepare the arguments to loop over.
	## See if we can use TQDM for a progress bar.
	try:
		import tqdm
		has_tqdm = True
	except:
		has_tqdm = False

	test_for_loop_args = zip(uniform_rngs, shapes, dims)
	if use_multiprocessing:		
		if has_tqdm:
			results = list(tqdm.tqdm(mp.Pool(mp.cpu_count()).imap(__mp_args__, test_for_loop_args), total=len(uniform_rngs)))
		else:
			print(f"  Failed to import tqdm (has_tqdm: {has_tqdm}). Running tests without a progress bar.")
			results = list(mp.Pool(mp.cpu_count()).starmap(__run_args__, test_for_loop_args))
	else:
		if has_tqdm:
			results = [__run_args__(*args) for args in tqdm.tqdm(list(zip(uniform_rngs, shapes, dims)))]
		else:
			print(f"  Failed to import tqdm (has_tqdm: {has_tqdm}). Running tests without a progress bar.")
			results = [__run_args__(*args) for args in zip(uniform_rngs, shapes, dims)]

	
	##  Process the results.
	average_results = {} ## Find the average of the results
	worst_results = {}  ## Find the worst of the results
	for result in results:
		for (k, v) in result.items():
			for (name_of_test_type, test_value) in v.items():
				flat_key = f"{k}.{name_of_test_type}"
				average_results[flat_key] = average_results.get(flat_key, []) + [test_value]
				if k != "stats":
					worst_results[flat_key] = max(worst_results.get(flat_key, 0), test_value)
	average_results = {f"average_{k}": np.mean(np.array(v)) for k, v in average_results.items()}
	worst_results = {f"worst_{k}": np.mean(np.array(v)) for k, v in worst_results.items()}

	## Print out results
	print(f"\n" * 3, end='',)
	print(f"*" * 60,)
	import pprint
	pprint.pp(average_results)
	print(f"\n" * 3, end='',)
	pprint.pp(worst_results)
	print(f"*" * 60,)

if __name__ == "__main__":
	__run_tests__()