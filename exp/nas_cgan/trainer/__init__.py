import torch.nn as nn

def get_trainer_dict(trainer_type):
  from . import (train_cnn_macro, cgan_cbn)

  trainer_dict = {
    'search_cgan_gen_cifar10': train_cnn_macro.Trainer,
    'retrain_cgan_gen_cifar10': train_cnn_macro.Trainer,

    'search_cgan_gen_cifar10_v1': train_cnn_macro.Trainer,
    'retrain_cgan_gen_cifar10_v1': train_cnn_macro.Trainer,

    'search_cgan_gen_cifar10_v2': train_cnn_macro.Trainer,
    'retrain_cgan_gen_cifar10_v2': train_cnn_macro.Trainer,

    'search_cgan_gen_cifar10_v3': cgan_cbn.Trainer,
    'retrain_cgan_gen_cifar10_v3': cgan_cbn.Trainer,

    'search_cgan_gen_cifar10_v4': train_cnn_macro.Trainer,
    'retrain_cgan_gen_cifar10_v4': train_cnn_macro.Trainer,
    'evaluate_cgan_gen_cifar10_v4': train_cnn_macro.Trainer,

    'search_cgan_gen_cifar10_v5': cgan_cbn.Trainer,
    'retrain_cgan_gen_cifar10_v5': cgan_cbn.Trainer,

    'retrain_cgan_gen_stl10_v4': train_cnn_macro.Trainer,

    'train_cgan_gen_mnist_v4': train_cnn_macro.Trainer,
    'train_cgan_gen_mnist_svhn_v4': train_cnn_macro.Trainer,

    'search_cgan_gen_cifar10_v6': train_cnn_macro.Trainer,
    'retrain_cgan_gen_cifar10_v6': train_cnn_macro.Trainer,
    'retrain_cgan_gen_cifar10_v6_cbn': train_cnn_macro.Trainer,
    'search_cgan_gen_stl10_v6_debug': train_cnn_macro.Trainer,
    'search_cgan_gen_stl10_v6': train_cnn_macro.Trainer,
    'retrain_cgan_gen_stl10_v6': train_cnn_macro.Trainer,
    'retrain_cgan_gen_stl10_v6_cbn': train_cnn_macro.Trainer,

    'retrain_cgan_gen_celeba_attr_v6': train_cnn_macro.Trainer,
    'retrain_cgan_gen_celeba_attr_v6_cbn': train_cnn_macro.Trainer,
    'train_cgan_gen_mnist_svhn_v6': train_cnn_macro.Trainer,
    'train_cgan_gen_mnist_svhn_v6_cbn': train_cnn_macro.Trainer,

    'search_cgan_gen_cifar10_v7': train_cnn_macro.Trainer,
    'retrain_cgan_gen_cifar10_v7': train_cnn_macro.Trainer,

    'search_cgan_gen_cifar10_v8': train_cnn_macro.Trainer,
    'retrain_cgan_gen_cifar10_v8': train_cnn_macro.Trainer,
  }
  return trainer_dict[trainer_type]

imsize_dict = {
  'C10': 32,
  'stl10': 48,
  'mnist': 32,
  'mnist_svhn': 32,
  'celeba_attr': 64,
}

nclass_dict = {
  'C10': 10,
  'stl10': 1000,
  'mnist': 10,
  'mnist_svhn': 20,
  'celeba_attr': 8,
}

activation_dict = {'inplace_relu': nn.ReLU(inplace=True),
                   'relu': nn.ReLU(inplace=False),
                   'ir': nn.ReLU(inplace=True),}


arcs_dict = {

}