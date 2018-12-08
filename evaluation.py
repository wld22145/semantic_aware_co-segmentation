import torch
import torchvision
from torchvision.models import vgg16
from torchvision.transforms import Compose, CenterCrop, Normalize, Scale, Resize, ToTensor, ToPILImage
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import Adam
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.optim as optim
import skimage.io as io

import time

import numpy as np
import glob
import cv2
import itertools
import PIL.Image as Image
import argparse

from datasets import coseg_val_dataset, coseg_train_dataset
import model_ca
import model_fca
import model_csa
import model_fcsa
import model_inception
import model_self
import model_blank
import model_resnet
import model_vgg
import model_vggbn




# input arguments
parser = argparse.ArgumentParser(description='Attention Based Co-segmentation')
parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=0.0005,
                    help='weight decay value')
parser.add_argument('--gpu_ids', default=[0,1,2,3], help='a list of gpus')
parser.add_argument('--num_worker', default=4, help='numbers of worker')
parser.add_argument('--batch_size', default=16, type=int, help='bacth size')
parser.add_argument('--epoches', default=5, help='epoches')

parser.add_argument('--model', default="ca", help='model structure')
parser.add_argument('--filename',help="filename of the model to test")

parser.add_argument('--train_data', default="Datasets/PascalVoc/image/", help='training data directory')
parser.add_argument('--val_data', default="Datasets/PascalVoc/image/", help='validation data directory')
parser.add_argument('--train_txt', default="Datasets/PascalVoc/colabel/train.txt", help='training image pair names txt')
parser.add_argument('--val_txt', default="Datasets/PascalVoc/colabel/val1600.txt", help='validation image pair names txt')
parser.add_argument('--train_label', default="Datasets/PascalVoc/colabel/train/", help='training label directory')
parser.add_argument('--val_label', default="Datasets/PascalVoc/colabel/val/", help='validation label directory')
parser.add_argument('--model_path', default="model_path/", help='model saving directory')

args = parser.parse_args()


# let the label pixels =1 if it >0
class Relabel:
    def __call__(self, tensor):
        assert isinstance(
            tensor, torch.LongTensor), 'tensor needs to be LongTensor'
        tensor[tensor > 0] = 1
        return tensor

# numpy -> tensor


class ToLabel:
    def __call__(self, image):
        return torch.from_numpy(np.array(image)).long()



class Trainer:
    def __init__(self):

        self.args = args


        self.input_transform = Compose([Resize((512, 512)), ToTensor(
        ), Normalize([.485, .456, .406], [.229, .224, .225])])
        self.label_transform = Compose(
            [Resize((512, 512)), CenterCrop(512), ToLabel(), Relabel()])


        self.net = self.parse_model().cuda()
        # self.net = nn.DataParallel(self.net, device_ids=self.args.gpu_ids)
        self.net = nn.DataParallel(self.net)

        self.val_data_loader = DataLoader(coseg_val_dataset(self.args.val_data, self.args.val_label, self.args.val_txt, self.input_transform, self.label_transform),
                                          num_workers=self.args.num_worker, batch_size=self.args.batch_size, shuffle=False)
        self.optimizer = optim.Adam(
            self.net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.loss_func = nn.CrossEntropyLoss()

    def parse_model(self):
        if self.args.model == "ca":
            return model_ca.model()
        elif self.args.model == "fca":
            return model_fca.model()
        elif self.args.model == "csa":
            return model_csa.model()
        elif self.args.model == "fcsa":
            return model_fcsa.model()
        elif self.args.model == "inception":
            return model_inception.model()
        elif self.args.model == "blank":
            return model_blank.model()
        elif self.args.model == "self":
            return model_self.model()
        elif self.args.model == "vgg":
            return model_vgg.model()
        elif self.args.model == "vggbn":
            return model_vggbn.model()
        elif self.args.model == "resnet":
            return model_resnet.model()
        else:
            print("cannot parse model's name")
            return None

    def pixel_accuracy(self, output, label):
        correct = len(output[output == label])
        wrong = len(output[output != label])
        return correct, wrong

    def jaccard(self, output, label):
        temp = output[label == 1]
        i = len(temp[temp == 1])
        temp = output + label
        u = len(temp[temp > 0])
        return i, u

    def precision(self, output, label):
        temp = output[label == 1]
        tp = len(temp[temp == 1])
        p = len(output[output > 0])
        return tp, p

    def evaluate(self, net, epoch):
        self.net.eval()
        correct = 0
        wrong = 0
        intersection = 0
        union = 0
        true_positive = 0
        positive = 1
        for i, (image1, image2, label1, label2) in enumerate(self.val_data_loader):

            image1, image2, label1, label2 = image1.cuda(
            ), image2.cuda(), label1.cuda(), label2.cuda()
            output1, output2 = self.net(image1, image2)
            output1 = torch.argmax(output1, dim=1)
            output2 = torch.argmax(output2, dim=1)
            # eval output1
            c, w = self.pixel_accuracy(output1, label1)
            correct += c
            wrong += w

            i, u = self.jaccard(output1, label1)
            intersection += i
            union += u

            tp, p = self.precision(output1, label1)
            true_positive += tp
            positive += p
            # eval output2
            c, w = self.pixel_accuracy(output2, label2)
            correct += c
            wrong += w

            i, u = self.jaccard(output2, label2)
            intersection += i
            union += u

            tp, p = self.precision(output2, label2)
            true_positive += tp
            positive += p

        print("pixel accuracy: {} correct: {}  wrong: {}".format(
            correct / (correct + wrong), correct, wrong))
        print("precision: {} true_positive: {} positive: {}".format(
            true_positive / positive, true_positive, positive))
        print("jaccard score: {} intersection: {} union: {}".format(
            intersection / union, intersection, union))
        self.net.train()
        return correct / (correct + wrong), intersection / union, true_positive / positive

    def test(self):
        print("testing ",self.args.filename)
        self.net.load_state_dict(torch.load(self.args.filename))
        print("model loaded")
        print("testing......")
        test_start_time = time.time()
        acc, jac, pre = self.evaluate(self.net, self.args.epoches)
        test_finish_time = time.time()
        print("test time consumption: ", test_finish_time - test_start_time)

if __name__ == "__main__":
    trainer = Trainer()
    trainer.test()
