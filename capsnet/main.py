import argparse
import numpy as np
import os

from capsnet.data.dataset import load_mnist
from capsnet.models.capsnet import CapsNet, test, train


def main():
    parser = argparse.ArgumentParser(description="Capsule Network")
    parser.add_argument('--epochs', default=50, type=int, help="Num of epochs to train")
    parser.add_argument('--batch_size', default=100, type=int, help="Batch size")
    parser.add_argument('--lr', default=0.001, type=float, help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float, help="Learning rate decay")
    parser.add_argument('--lam_recon', default=0.392, type=float, help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int, help="Number of iterations used in routing algorithm")
    parser.add_argument('--save_dir', default='./result')

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    (x_train, y_train), (x_test, y_test) = load_mnist()
    model, eval_model = CapsNet(input_shape=x_train.shape[1:], n_class=len(np.unique(np.argmax(y_train, 1))),
                                routings=args.routings)
    model.summary()
    train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
    test(model=eval_model, data=(x_test, y_test), args=args)


if __name__ == '__main__':
    main()
