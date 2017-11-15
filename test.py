"""
A script to predict nodules using conv net model and for analysis of results
"""
import torch
import torch.nn as nn
import pickle
import pandas as pd
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import itertools
from torchvision import models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import data

hdfs_file = '../data/test.h5'


def create_mosaic(image, nrows, ncols):
    """
    Tiles all the layers in nrows x ncols
    Args:
    ------
    image = 3d numpy array of M * N * number of filters dimensions
    nrows = integer representing number of images in a row
    ncol = integer representing number of images in a column
    returns formatted image
    """

    M = image.shape[1]
    N = image.shape[2]

    npad = ((0, 0), (1, 1), (1, 1))
    image = np.pad(image, pad_width=npad, mode='constant', \
                   constant_values=0)
    M += 2
    N += 2
    image = image.reshape(nrows, ncols, M, N)
    image = np.transpose(image, (0, 2, 1, 3))
    image = image.reshape(M * nrows, N * ncols)
    return image


def format_image(image, num_images):
    """
    Formats images
    """
    idxs = np.random.choice(image.shape[0], num_images)
    M = image.shape[1]
    N = image.shape[2]
    imagex = np.squeeze(image[idxs, :, :, :])
    print (imagex.shape)
    return imagex


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Purples):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    # plt.grid('off')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_predictions(images, filename):
    """
    Plots the predictions mosaic
    """
    imagex = format_image(images, 4)
    mosaic = create_mosaic(imagex, 2, 2)
    plt.figure(figsize=(12, 12))
    plt.imshow(mosaic, cmap='gray')
    plt.axis('off')
    plt.savefig(filename + '.png', bbox_inches='tight')


def get_predictions(model, loader, use_gpu, criterion, dset_size):
    """
    Args:
    ------
    Given hdfs file of X_test_images and Y_test_labels

    returns:
    --------
    predictions: probability values for each class
    label_predictions: returns predicted classes
    """
    running_loss = 0.0
    running_corrects = 0
    predictions = []
    label_predictions = []
    Y_labels = []

    for data in loader:
        # get the inputs
        inputs, labels = data

        # wrap them in variable
        if use_gpu:
            inputs, labels = Variable(inputs, volatile=True).cuda(), \
                             Variable(labels, volatile=True).cuda()
        else:
            inputs, labels = Variable(inputs, volatile=True), \
                             Variable(labels, volatile=True)

        # forward
        outputs = model(inputs)

        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        # statistics
        running_loss += loss.data[0]
        running_corrects += torch.sum(preds == labels.data)
        predictions += (nn.Softmax()(outputs)).data.tolist()
        label_predictions += preds.tolist()
        Y_labels += labels.data.tolist()

    epoch_loss = running_loss / dset_size
    epoch_acc = running_corrects / dset_size
    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        'test', epoch_loss, epoch_acc))

    return np.asarray(predictions), np.asarray(label_predictions), np.asarray(Y_labels)


def get_roc_curve(Y_test_labels, predictions):
    """
    Args:
    -------
    hdfs datasets: Y_test_labels and predictions

    Returns:
    --------
    fpr: false positive Rate
    tpr: true posiive Rate
    roc_auc: area under the curve value
    """
    fpr, tpr, thresholds = roc_curve(Y_test_labels, predictions[:, 1], pos_label=1)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def get_metrics(Y_test_labels, label_predictions):
    """
    Args:
    -----
    Y_test_labels, label_predictions
    Returns:
    --------
    precision, recall and specificity values and cm
    """
    cm = confusion_matrix(Y_test_labels[:, 1], label_predictions[:, 1])

    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TP = cm[1][1]

    precision = TP * 1.0 / (TP + FP)
    recall = TP * 1.0 / (TP + FN)
    specificity = TN * 1.0 / (TN + FP)

    return precision, recall, specificity, cm


def plot_roc_curve(fpr, tpr, roc_auc):
    """
    Plots ROC curve
    Args:
    -----
    FPR, TPR and AUC
    """
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='(AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.axis('equal')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig('roc1.png', bbox_inches='tight')


def main():

    # dataset setting
    transform = transforms.Compose([
        transforms.Scale((224,224)),
        data.ToTensor()
        ])
    dset = data.Luna2DData('test',transform)
    dset_loader = DataLoader(dset, batch_size=96,
                                  shuffle=False, num_workers=8)
    dset_size = len(dset)
    dset_classes = dset.classes

    # gpu option
    use_gpu = torch.cuda.is_available()

    # get module
    model = models.alexnet()
    model.train = False

    # replace final layer
    mod = list(model.classifier.children())
    mod.pop()
    mod.append(torch.nn.Linear(4096, 2))
    new_classifier = torch.nn.Sequential(*mod)
    model.classifier = new_classifier

    # load weight
    model.load_state_dict(torch.load('weights.save'))

    # load model to GPU
    if use_gpu:
        model = model.cuda()

    # choose critetion
    criterion = nn.CrossEntropyLoss()

    # get pred, label_pred, Y_label
    predictions, label_predictions, Y_labels = \
        get_predictions(model,dset_loader,use_gpu,criterion,dset_size)

    # calc fp, tp, auc
    fpr, tpr, roc_auc = get_roc_curve(Y_labels, predictions)
    print (roc_auc)

    return

    plot_roc_curve(fpr, tpr, roc_auc)

    precision, recall, specificity, cm = \
        get_metrics(Y_labels, label_predictions)

    print(precision, recall, specificity)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cm, classes=['no-nodule', 'nodule'], \
                          title='Confusion matrix')
    plt.savefig('confusion_matrix.png', bbox_inches='tight')

    # Plot all inputs representing True Positives, FP, FN, TN
    TP_images = X_test_images[(Y_test_labels[:, 1] == 1) & (label_predictions[:, 1] == 1), :, :, :]
    FP_images = X_test_images[(Y_test_labels[:, 1] == 0) & (label_predictions[:, 1] == 1), :, :, :]
    TN_images = X_test_images[(Y_test_labels[:, 1] == 0) & (label_predictions[:, 1] == 0), :, :, :]
    FN_images = X_test_images[(Y_test_labels[:, 1] == 1) & (label_predictions[:, 1] == 0), :, :, :]

    ## Choose 16 images randomly
    plot_predictions(TP_images, 'preds_tps')
    plot_predictions(TN_images, 'preds_tns')
    plot_predictions(FN_images, 'preds_fns')
    plot_predictions(FP_images, 'preds_fps')


if __name__ == "__main__":
    main()