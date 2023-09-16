# My adjustment

- [x] Put FF algorithm into `Models/ff_model.py`
- [x] Put customized dataloader into `Data/customized_MNIST.py`
- [x] You can now train/test by runing `python my_train.py`
- [x] In `python my_train.py` (as well as `Models/ff_model.py`) you can choose a new option to sampling the negative samples (`sampling_neg = 'rand' | 'perm'`). 
    - `rand` is the new option that we added. It turns out that this process reduce the error when using low batch size, e.g. batch size = 128,512, 1024 from more than 0.8 to ~ 0.14, 0.10, 0.08.
    - `perm` is from the original implementation, which is kept as a reference  
- [ ] Provide some testing across different variations/settings. 

### My observations : 

- More layers ==> Higher error 
- Lower batch size ==> Higher error (I observe that this is a big problem in the original implementation, but we add a new option (`rand`) which fix most of this.)
- lower sub_epoch  ==> Higher error 
- Higher learning rate ==> Higher error 
- **As the reference,** the orignal implementation is still kept as `main.py` ... 

# pytorch_forward_forward
Implementation of forward-forward (FF) training algorithm - an alternative to back-propagation
---

Below is my understanding of the FF algorithm presented at [Geoffrey Hinton's talk at NeurIPS 2022](https://www.cs.toronto.edu/~hinton/FFA13.pdf).\
The conventional backprop computes the gradients by successive applications of the chain rule, from the objective function to the parameters. FF, however, computes the gradients locally with a local objective function, so there is no need to backpropagate the errors.

![](./imgs/BP_vs_FF.png)

The local objective function is designed to push a layer's output to values larger than a threshold for positive samples and to values smaller than a threshold for negative samples.

A positive sample $s$ is a real datapoint with a large $P(s)$ under the training distribution.\
A negative sample $s'$ is a fake datapoint with a small $P(s')$ under the training distribution.

![](./imgs/layer.png)

Among the many ways of generating the positive/negative samples, for MNIST, we have:\
Positive sample $s = merge(x, y)$, the image and its label\
Negative sample $s' = merge(x, y_{random})$, the image and a random label

![](./imgs/pos_neg.png)

After training all the layers, to make a prediction for a test image $x$, we find the pair $s = (x, y)$ for all $0 \leq y < 10$ that maximizes the network's overall activation.

With this implementation, the training and test errors on MNIST are:
```python
> python main.py
train error: 0.06754004955291748
test error: 0.06840002536773682
```
