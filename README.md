# Pytorch-WGAN-GP-mnist

This is a private Pytorch practice for the WGAN-GP implementation.

By the way, I use the MNIST dataset as our training data.

Hope this code can help you who find this repo. :)


## How to use

1. requirements

```
torch=1.10.0
torchvision=0.11.1
```

2. run the python script.

```
python main.py --lr 0.00001 \
               --batch_size 64 \
               --epochs 100 \
               --optimizer adam \
               --normalize \
               --log
```

3. open another terminal and use Tensorboard to track the training process.

```
tensorboard --logdir logs
```

4. The codes may be updated. To be continued.