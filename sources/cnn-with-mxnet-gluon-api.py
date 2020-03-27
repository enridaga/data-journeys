
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import mxnet as mx # mxnet module
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, utils as gutils, nn
import time
import math
train = pd.read_csv('../input/Kannada-MNIST/train.csv')
test = pd.read_csv('../input/Kannada-MNIST/test.csv')

train.shape, test.shape
split = int(train.shape[0] * 0.9) # split the train data into train and valid with a ratio
X_train = nd.array(train.iloc[:split, 1:].values.astype(dtype=np.uint8)).reshape(split, 28, 28, 1)
y_train = nd.array(train.iloc[:split, 0].values.astype(dtype=np.int32)).reshape(split,)
X_valid = nd.array(train.iloc[split:, 1:].values.astype(dtype=np.uint8)).reshape(train.shape[0] - split, 28, 28, 1)
y_valid = nd.array(train.iloc[split:, 0].values.astype(dtype=np.int32)).reshape(train.shape[0] - split,)
X_train.shape, X_valid.shape, y_train.shape, y_valid.shape
# construct the array dataset
train_data = gdata.dataset.ArrayDataset(X_train, y_train)
valid_data = gdata.dataset.ArrayDataset(X_valid, y_valid)

# batch data
batch_size = 1000
# transform data for a large dataset
transform_train = gdata.vision.transforms.Compose([
    gdata.vision.transforms.Resize(35),
    gdata.vision.transforms.RandomResizedCrop(28, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
    gdata.vision.transforms.RandomFlipLeftRight(),
    gdata.vision.transforms.ToTensor(),
    gdata.vision.transforms.Normalize(0.0819, 0.2412)
])
transform_valid = gdata.vision.transforms.Compose([
    gdata.vision.transforms.ToTensor(),
    gdata.vision.transforms.Normalize(0.0819, 0.2412)
])
train_data = train_data.transform_first(transform_train)
valid_data = valid_data.transform_first(transform_valid)
train_iter = gdata.DataLoader(train_data, batch_size, shuffle=True, num_workers=0)
valid_iter = gdata.DataLoader(valid_data, batch_size, shuffle=False, num_workers=0)
# build a CNN
def get_net(ctx):
    net = nn.HybridSequential()
    net.add(
        # first
        nn.Conv2D(64, kernel_size=3, padding=1),
        nn.BatchNorm(),
        nn.LeakyReLU(0.1),
        nn.Conv2D(64, kernel_size=3, padding=1),
        nn.BatchNorm(),
        nn.LeakyReLU(0.1),
        nn.Conv2D(64, kernel_size=5, padding=2),
        nn.BatchNorm(),
        nn.LeakyReLU(0.1),
        nn.MaxPool2D(pool_size=2),
        nn.Dropout(0.2),
        # second
        nn.Conv2D(128, kernel_size=3, padding=1),
        nn.BatchNorm(),
        nn.LeakyReLU(0.1),
        nn.Conv2D(128, kernel_size=3, padding=1),
        nn.BatchNorm(),
        nn.LeakyReLU(0.1),
        nn.Conv2D(128, kernel_size=5, padding=2),
        nn.BatchNorm(),
        nn.LeakyReLU(0.1),
        nn.MaxPool2D(pool_size=2),
        nn.Dropout(0.2),
        # third
        nn.Conv2D(256, kernel_size=3, padding=1),
        nn.BatchNorm(),
        nn.LeakyReLU(0.1),
        nn.Conv2D(256, kernel_size=3, padding=1),
        nn.BatchNorm(),
        nn.LeakyReLU(0.1),
        nn.MaxPool2D(pool_size=2),
        nn.Dropout(0.2),
        # fourth
        nn.Flatten(),
        nn.Dense(256),
        nn.BatchNorm(),
        nn.LeakyReLU(0.1),
        nn.Dense(10)
    )
    net.initialize(ctx=ctx, init=init.Xavier())
    return net
def _get_batch(batch, ctx):
    """Return features and labels on ctx."""
    features, labels = batch
    if labels.dtype != features.dtype:
        labels = labels.astype(features.dtype)
    return (gutils.split_and_load(features, ctx),
            gutils.split_and_load(labels, ctx), features.shape[0])

def evaluate_accuracy(data_iter, net, ctx=[mx.cpu()]):
    """Evaluate accuracy of a model on the given data set."""
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc_sum, n = nd.array([0]), 0
    for batch in data_iter:
        features, labels, _ = _get_batch(batch, ctx)
        for X, y in zip(features, labels):
            y = y.astype('float32')
            acc_sum += (net(X).argmax(axis=1) == y).sum().copyto(mx.cpu())
            n += y.size
        acc_sum.wait_to_read()
    return acc_sum.asscalar() / n

def train(net, train_iter, valid_iter, split, batch_size, num_epochs, ctx):
    """Train the CNN"""
    iterations_per_epoch = math.ceil(split / batch_size)
    schedule = mx.lr_scheduler.FactorScheduler(step=20 * iterations_per_epoch, factor=0.1)
    rmsprop_optim = mx.optimizer.RMSProp(lr_scheduler=schedule)
    trainer = gluon.Trainer(net.collect_params(), optimizer=rmsprop_optim)
    loss = gloss.SoftmaxCrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            y = y.astype('float32').as_in_context(ctx)
            with autograd.record():
                y_hat  = net(X.as_in_context(ctx))
                l = loss(y_hat, y).sum()
            l.backward()
            trainer.step(batch_size)
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size
        time_s = "time %.2f sec" % (time.time() - start)
        if valid_iter is not None:
            valid_acc = evaluate_accuracy(valid_iter, net, ctx)
            epoch_s = ('epoch %d, loss %f, train acc %f, valid acc %f, ' %
                      (epoch + 1, train_l_sum / n, train_acc_sum / n, valid_acc))
        else:
            epoch_s = ('epoch %d, loss %f, train acc %f, ' %
                      (epoch + 1, train_l_sum / n, train_acc_sum / n))
        print(epoch_s + time_s + ', lr ' + str(trainer.learning_rate))
ctx, num_epochs = mx.gpu(0), 50
net = get_net(ctx)
net.hybridize()
train(net, train_iter, valid_iter, split, batch_size, num_epochs, ctx)
X_test = nd.array(test.iloc[:, 1:].values.astype(dtype=np.uint8)).reshape(test.shape[0], 28, 28, 1)
test_data = gdata.dataset.ArrayDataset(X_test)
test_data = test_data.transform_first(transform_valid)
test_iter = gdata.DataLoader(test_data, batch_size, shuffle=False, num_workers=0)

preds = []
for X in test_iter:
    y_hat = net(X.as_in_context(ctx))
    preds.extend(y_hat.argmax(axis=1).astype(int).asnumpy())
sample_submission = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')
sample_submission['label'] = pd.Series(preds)
submission = pd.concat([sample_submission['id'], sample_submission['label']], axis=1)
submission.to_csv('submission.csv', index=False)

