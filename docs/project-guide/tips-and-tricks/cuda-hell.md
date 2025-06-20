# Some tips and tricks to deal with the CUDA hell



These are some tips and tricks that may be helpful while dealing with CUDA.


## Install the cuda-toolkit

It is important to remember that one **must** install the cuda-toolkit, and set the CUDA_HOME environment variable, in order for some critical tools, such as `vLLM` to work. To do this, follow the instructions at: [CUDA installation guide on Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)


## Flush cuda memory from Python code

It is helpful to start all Python code that uses CUDA with the following mantra invocation:

```python

import torch, gc
gc.collect()
torch.cuda.empty_cache()
```

Despite this invocation, quite often it will not completely flush the cuda memory. In that case, a better invocation at the level of the Linux shell is:

```bash
nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 kill -9
```
This pearl of wisdom is gleaned from: [How to flush GPU memory using CUDA ](https://stackoverflow.com/questions/15197286/how-can-i-flush-gpu-memory-using-cuda-physical-reset-is-unavailable)

If you're still hitting unexpected memory errors or similar problems then try:

```bash
sudo fuser -v /dev/nvidia* | cut -d' ' -f2- | sudo xargs -n1 kill -9
```

### `nvtop` is your friend!

A very useful and visual tool to see what is happening in CUDA is to use the tool `nvtop`. Install it with the mantra:

```bash
sudo dnf install nvtop
```
(or its equivalent if you foolishly use anything other than redhat/centos/rocky-linux).

