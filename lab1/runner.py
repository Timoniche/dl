import os
import pathlib
import warnings

import numpy as np
import pytorch_lightning as pl

from pytorch_lightning.loggers import CSVLogger
from torch import nn
from scipy.io import loadmat

from lab1.classifier import Classifier
from lab1.datamodule import DataModule
from lab1.inception_v3_model import InceptionModelV3
from lab1.plotter_utils import plot_metrics


def run_inception_v3():
    current_dir = str(pathlib.Path().resolve())
    dataset_dir = current_dir + '/cars_dataset'

    # train_imgs = os.listdir(dataset_dir + '/cars_train')
    # test_imgs = os.listdir(dataset_dir + '/cars_test')

    cars_test_annos = loadmat(dataset_dir + '/cars_test_annos_withlabels_eval.mat')
    cars_train_annos = loadmat(dataset_dir + '/cars_train_annos.mat')
    # cars_meta = loadmat(dataset_dir + '/cars_meta.mat')

    # class_names = [meta[0] for meta in cars_meta['class_names'][0]]
    training_class_dict = {}
    test_class_dict = {}

    for ann in cars_train_annos['annotations'][0]:
        image, label = ann[-1][0], ann[-2][0][0] - 1
        training_class_dict[image] = label

    for ann in cars_test_annos['annotations'][0]:
        image, label = ann[-1][0], ann[-2][0][0] - 1
        test_class_dict[image] = label

    datamodule = DataModule(
        root=dataset_dir,
        image_size=299,  # ImageNet pretrained with (3 x 299 x 299)
        batch_size=1 << 6,
        train_class_dict=training_class_dict,
        test_class_dict=test_class_dict,
    )
    num_classes = datamodule.num_classes
    inception_v3 = InceptionModelV3(
        num_classes=num_classes,
        model_pretrained_path=None,
        classification_layer_path=None,
    )

    max_epochs = 1
    _run_model_on_datamodule(
        model=inception_v3,
        datamodule=datamodule,
        max_epochs=max_epochs,
        save_classification_layer=True,
        checkpoint_path=None
    )


def _run_model_on_datamodule(
        model: nn.Module,
        datamodule,
        max_epochs,
        save_classification_layer,
        checkpoint_path=None,
):
    warnings.filterwarnings("ignore")

    num_classes = model.num_classes
    classifier = Classifier(
        model=model,
        num_classes=num_classes,
    )
    current_dir = str(pathlib.Path().resolve())
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='auto',
        logger=[
            CSVLogger(
                save_dir=current_dir,
                name='lightning_logs',
            )
        ]
    )

    trainer.fit(
        classifier,
        datamodule=datamodule,
        ckpt_path=checkpoint_path,
    )

    logs_version_path = _lightning_logs_last_version_path()

    if save_classification_layer:
        model.save_classificator_layer(path=logs_version_path + '/classification_layer.pth')

    model.eval()
    _ = trainer.test(classifier, datamodule=datamodule)

    plot_metrics(logs_version_path + '/metrics.csv')


def _lightning_logs_last_version_path():
    current_dir = str(pathlib.Path().resolve())
    log_version = _max_log_version(current_dir)
    return current_dir + '/' + 'lightning_logs' + '/' + 'version_' + str(log_version)


def _max_log_version(root):
    dirs = os.listdir(root + '/lightning_logs')
    dirs = list(filter(lambda d: d.startswith('version_'), dirs))
    log_versions = list(map(lambda d: int(d.removeprefix('version_')), dirs))
    max_version = np.max(log_versions)

    return max_version


def main():
    run_inception_v3()


if __name__ == '__main__':
    main()
