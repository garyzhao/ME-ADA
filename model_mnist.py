from __future__ import print_function, absolute_import, division

import os
import torch
import numpy as np
from torch.autograd import Variable
from torch.optim import lr_scheduler

from models.lenet import LeNet5
from common.data_gen_MNIST import BatchImageGenerator, get_data_loaders
from common.utils import fix_all_seed, write_log, adam, sgd, compute_accuracy, entropy_loss


class ModelBaseline(object):
    def __init__(self, flags):

        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        self.setup(flags)
        self.setup_path(flags)
        self.configure(flags)

    def setup(self, flags):
        torch.backends.cudnn.deterministic = flags.deterministic
        print('torch.backends.cudnn.deterministic:', torch.backends.cudnn.deterministic)
        fix_all_seed(flags.seed)

        self.network = LeNet5(num_classes=flags.num_classes)
        self.network = self.network.cuda()

        print(self.network)
        print('flags:', flags)
        if not os.path.exists(flags.logs):
            os.makedirs(flags.logs)

        flags_log = os.path.join(flags.logs, 'flags_log.txt')
        write_log(flags, flags_log)

    def setup_path(self, flags):

        root_folder = 'data'
        data, data_loaders = get_data_loaders()

        seen_index = flags.seen_index
        self.train_data = data[seen_index]
        self.test_data = [x for index, x in enumerate(data) if index != seen_index]

        if not os.path.exists(flags.logs):
            os.makedirs(flags.logs)

        flags_log = os.path.join(flags.logs, 'path_log.txt')
        write_log(str(self.train_data), flags_log)
        write_log(str(self.test_data), flags_log)

        self.batImageGenTrain = BatchImageGenerator(flags=flags, stage='train', file_path=root_folder,
                                                    data_loader=data_loaders[seen_index], b_unfold_label=False)

        self.batImageGenTests = []
        for index, test_loader in enumerate(data_loaders):
            if index != seen_index:
                batImageGenTest = BatchImageGenerator(flags=flags, stage='test', file_path=root_folder,
                                                      data_loader=test_loader, b_unfold_label=False)
                self.batImageGenTests.append(batImageGenTest)

    def configure(self, flags):

        for name, param in self.network.named_parameters():
            print(name, param.size())

        self.optimizer = adam(parameters=self.network.parameters(),
                              lr=flags.lr,
                              weight_decay=flags.weight_decay)

        self.scheduler = lr_scheduler.StepLR(optimizer=self.optimizer, step_size=flags.step_size, gamma=0.1)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def train(self, flags):
        self.network.train()
        self.best_accuracy_test = -1

        for ite in range(flags.loops_train):

            self.scheduler.step(epoch=ite)

            # get the inputs and labels from the data reader
            total_loss = 0.0
            images_train, labels_train = self.batImageGenTrain.get_images_labels_batch()

            inputs, labels = torch.from_numpy(np.array(images_train, dtype=np.float32)), torch.from_numpy(
                np.array(labels_train, dtype=np.float32))

            # wrap the inputs and labels in Variable
            inputs, labels = Variable(inputs, requires_grad=False).cuda(), \
                             Variable(labels, requires_grad=False).long().cuda()

            # forward with the adapted parameters
            outputs, _ = self.network(x=inputs)

            # loss
            loss = self.loss_fn(outputs, labels)

            # init the grad to zeros first
            self.optimizer.zero_grad()

            # backward your network
            loss.backward()

            # optimize the parameters
            self.optimizer.step()

            if ite < 500 or ite % 500 == 0:
                print(
                    'ite:', ite, 'total loss:', loss.cpu().item(), 'lr:',
                    self.scheduler.get_lr()[0])

            flags_log = os.path.join(flags.logs, 'loss_log.txt')
            write_log(str(loss.item()), flags_log)

            if ite % flags.test_every == 0 and ite is not 0:
                self.test_workflow(self.batImageGenTests, flags, ite)

    def test_workflow(self, batImageGenTests, flags, ite):

        accuracies = []
        for count, batImageGenTest in enumerate(batImageGenTests):
            accuracy_test = self.test(batImageGenTest=batImageGenTest, ite=ite,
                                      log_dir=flags.logs, log_prefix='test_index_{}'.format(count))
            accuracies.append(accuracy_test)

        mean_acc = np.mean(accuracies)

        if mean_acc > self.best_accuracy_test:
            self.best_accuracy_test = mean_acc

            f = open(os.path.join(flags.logs, 'best_test.txt'), mode='a')
            f.write('ite:{}, best test accuracy:{}\n'.format(ite, self.best_accuracy_test))
            f.close()

            if not os.path.exists(flags.model_path):
                os.makedirs(flags.model_path)

            outfile = os.path.join(flags.model_path, 'best_model.tar')
            torch.save({'ite': ite, 'state': self.network.state_dict()}, outfile)

    def test(self, batImageGenTest, ite, log_prefix, log_dir='logs/'):

        # switch on the network test mode
        self.network.eval()

        images_test = batImageGenTest.images
        labels_test = batImageGenTest.labels

        threshold = 50
        if len(images_test) > threshold:

            n_slices_test = int(len(images_test) / threshold)
            indices_test = []
            for per_slice in range(n_slices_test - 1):
                indices_test.append(int(len(images_test) * (per_slice + 1) / n_slices_test))
            test_image_splits = np.split(images_test, indices_or_sections=indices_test)

            # Verify the splits are correct
            test_image_splits_2_whole = np.concatenate(test_image_splits)
            assert np.all(images_test == test_image_splits_2_whole)

            # split the test data into splits and test them one by one
            test_image_preds = []
            for test_image_split in test_image_splits:
                images_test = Variable(torch.from_numpy(np.array(test_image_split, dtype=np.float32))).cuda()
                tuples = self.network(images_test)

                predictions = tuples[-1]['Predictions']
                predictions = predictions.cpu().data.numpy()
                test_image_preds.append(predictions)

            # concatenate the test predictions first
            predictions = np.concatenate(test_image_preds)
        else:
            images_test = Variable(torch.from_numpy(np.array(images_test, dtype=np.float32))).cuda()
            tuples = self.network(images_test)

            predictions = tuples[-1]['Predictions']
            predictions = predictions.cpu().data.numpy()

        accuracy = compute_accuracy(predictions=predictions, labels=labels_test)
        print('----------accuracy test----------:', accuracy)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        f = open(os.path.join(log_dir, '{}.txt'.format(log_prefix)), mode='a')
        f.write('ite:{}, accuracy:{}\n'.format(ite, accuracy))
        f.close()

        # switch on the network train mode
        self.network.train()

        return accuracy


class ModelADA(ModelBaseline):
    def __init__(self, flags):
        super(ModelADA, self).__init__(flags)

    def configure(self, flags):
        super(ModelADA, self).configure(flags)
        self.dist_fn = torch.nn.MSELoss()

    def maximize(self, flags):
        self.network.eval()

        images_train, labels_train = self.batImageGenTrain.images, self.batImageGenTrain.labels
        images, labels = [], []

        for start, end in zip(range(0, len(labels_train), flags.batch_size),
                              range(flags.batch_size, len(labels_train), flags.batch_size)):
            inputs, targets = torch.from_numpy(
                np.array(images_train[start:end], dtype=np.float32)), torch.from_numpy(
                np.array(labels_train[start:end], dtype=np.float32))

            # wrap the inputs and labels in Variable
            inputs, targets = Variable(inputs, requires_grad=False).cuda(), \
                              Variable(targets, requires_grad=False).long().cuda()

            inputs_embedding = self.network(x=inputs)[-1]['Embedding'].detach().clone()
            inputs_embedding.requires_grad_(False)

            inputs_max = inputs.detach().clone()
            inputs_max.requires_grad_(True)
            optimizer = sgd(parameters=[inputs_max], lr=flags.lr_max)

            for ite_max in range(flags.loops_adv):
                tuples = self.network(x=inputs_max)

                # loss
                loss = self.loss_fn(tuples[0], targets) - flags.gamma * self.dist_fn(
                    tuples[-1]['Embedding'], inputs_embedding)

                # init the grad to zeros first
                self.network.zero_grad()
                optimizer.zero_grad()

                # backward your network
                (-loss).backward()

                # optimize the parameters
                optimizer.step()

                flags_log = os.path.join(flags.logs, 'max_loss_log.txt')
                write_log('ite_adv:{}, {}'.format(ite_max, loss.item()), flags_log)

            inputs_max = inputs_max.detach().clone().clamp(min=0.0, max=1.0)
            images.append(inputs_max.cpu().numpy())
            labels.append(targets.cpu().numpy())

        return np.concatenate(images), np.concatenate(labels)

    def train(self, flags):
        counter_k = 0
        self.best_accuracy_test = -1

        for ite in range(flags.loops_train):
            if ((ite + 1) % flags.loops_min == 0) and (counter_k < flags.k):  # if T_min iterations are passed
                print('Generating adversarial images [iter {}]'.format(counter_k))
                images, labels = self.maximize(flags)
                self.batImageGenTrain.images = np.concatenate((self.batImageGenTrain.images, images))
                self.batImageGenTrain.labels = np.concatenate((self.batImageGenTrain.labels, labels))
                self.batImageGenTrain.shuffle()
                counter_k += 1

            self.network.train()
            self.scheduler.step(epoch=ite)

            # get the inputs and labels from the data reader
            images_train, labels_train = self.batImageGenTrain.get_images_labels_batch()

            inputs, labels = torch.from_numpy(np.array(images_train, dtype=np.float32)), torch.from_numpy(
                np.array(labels_train, dtype=np.float32))

            # wrap the inputs and labels in Variable
            inputs, labels = Variable(inputs, requires_grad=False).cuda(), \
                             Variable(labels, requires_grad=False).long().cuda()

            # forward with the adapted parameters
            outputs, _ = self.network(x=inputs)

            # loss
            loss = self.loss_fn(outputs, labels)

            # init the grad to zeros first
            self.optimizer.zero_grad()

            # backward your network
            loss.backward()

            # optimize the parameters
            self.optimizer.step()

            if ite < 500 or ite % 500 == 0:
                print(
                    'ite:', ite, 'total loss:', loss.cpu().item(), 'lr:',
                    self.scheduler.get_lr()[0])

            flags_log = os.path.join(flags.logs, 'loss_log.txt')
            write_log(str(loss.item()), flags_log)

            if ite % flags.test_every == 0 and ite is not 0:
                self.test_workflow(self.batImageGenTests, flags, ite)


class ModelMEADA(ModelADA):
    def __init__(self, flags):
        super(ModelMEADA, self).__init__(flags)

    def maximize(self, flags):
        self.network.eval()

        images_train, labels_train = self.batImageGenTrain.images, self.batImageGenTrain.labels
        images, labels = [], []

        for start, end in zip(range(0, len(labels_train), flags.batch_size),
                              range(flags.batch_size, len(labels_train), flags.batch_size)):
            inputs, targets = torch.from_numpy(
                np.array(images_train[start:end], dtype=np.float32)), torch.from_numpy(
                np.array(labels_train[start:end], dtype=np.float32))

            # wrap the inputs and labels in Variable
            inputs, targets = Variable(inputs, requires_grad=False).cuda(), \
                              Variable(targets, requires_grad=False).long().cuda()

            inputs_embedding = self.network(x=inputs)[-1]['Embedding'].detach().clone()
            inputs_embedding.requires_grad_(False)

            inputs_max = inputs.detach().clone()
            inputs_max.requires_grad_(True)
            optimizer = sgd(parameters=[inputs_max], lr=flags.lr_max)

            for ite_max in range(flags.loops_adv):
                tuples = self.network(x=inputs_max)

                # loss
                loss = self.loss_fn(tuples[0], targets) + flags.eta * entropy_loss(tuples[0]) - \
                       flags.gamma * self.dist_fn(tuples[-1]['Embedding'], inputs_embedding)

                # init the grad to zeros first
                self.network.zero_grad()
                optimizer.zero_grad()

                # backward your network
                (-loss).backward()

                # optimize the parameters
                optimizer.step()

                flags_log = os.path.join(flags.logs, 'max_loss_log.txt')
                write_log('ite_adv:{}, {}'.format(ite_max, loss.item()), flags_log)

            inputs_max = inputs_max.detach().clone().clamp(min=0.0, max=1.0)
            images.append(inputs_max.cpu().numpy())
            labels.append(targets.cpu().numpy())

        return np.concatenate(images), np.concatenate(labels)
