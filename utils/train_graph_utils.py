from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import logging
import os
import time
import warnings
import torch
from torch import nn
from torch import optim
from torch_geometric.data import DataLoader
import model_node
import datasets
from utils.save import Save_Tool
from utils.freeze import set_freeze_by_id
import model_graph

class train_utils(object):
    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir

        # 创建 TensorBoard SummaryWriter //hjz
        self.writer = SummaryWriter(
            log_dir='E:/001 自行查阅/素材6_GNN/GNN_时序信号混合/GNNforIFD/GNNforIFD/checkpoint/tensorboard/logsseu_node_1')

    def visualize_features_with_tsne(self):
        # 仅选择一批数据进行可视化
        for data in self.dataloaders['train']:
            inputs = data.to(self.device)
            labels = inputs.y.detach().cpu().numpy()

            # 假设你的模型有一个方法来获取最后的特征表示，这里我们直接使用inputs作为特征，实际使用中应替换为模型输出
            features = inputs.x.detach().cpu().numpy()

            # 使用t-SNE进行降维到3维空间
            tsne = TSNE(n_components=3, perplexity=30, n_iter=300)
            features_3d = tsne.fit_transform(features)

            # 可视化
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')

            # 为每个标签分配不同的颜色
            scatter = ax.scatter(features_3d[:, 0], features_3d[:, 1], features_3d[:, 2], c=labels, cmap='rainbow')
            legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
            ax.add_artist(legend1)

            plt.title('t-SNE visualization of node features before training')
            plt.show()

            break  # 只对第一批数据进行操作

    def setup(self):
        args = self.args

        # gpu or cpu
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))

        # 加载数据集
        Dataset = getattr(datasets, args.data_name)
        self.datasets = {}
        self.datasets['train'], self.datasets['val'] = Dataset(args.sample_length, args.data_dir, args.input_type,
                                                               args.task).data_prepare()
        self.dataloaders = {x: DataLoader(self.datasets[x], batch_size=args.batch_size,
                                          shuffle=(True if x == 'train' else False),
                                          num_workers=args.num_workers,
                                          pin_memory=(True if self.device == 'cuda' else False))
                            for x in ['train', 'val']}

        # 定义模型
        input_type = args.input_type
        if input_type == "TD":
            feature = args.sample_length
        elif input_type == "FD":
            # feature = int(args.sample_length / 2)  # FFT函数 对应折半
            # feature = int(args.sample_length)
            feature = int(args.sample_length)
        elif input_type == "other":
            feature = 1
        else:
            print("输入类型有误!")

        if args.task == 'Node':
            self.model = getattr(model_node, args.model_name)(feature=feature, out_channel=Dataset.num_classes)
        elif args.task == 'Graph':
            self.model = getattr(model_graph, args.model_name)(feature=feature, out_channel=Dataset.num_classes,
                                                               pooltype=args.pooltype)
        else:
            print('任务类型有误!')

        if args.layer_num_last != 0:
            set_freeze_by_id(self.model, args.layer_num_last)
        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model)

        # 选择优化器
        if args.opt == 'sgd':
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                       momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.opt == 'adam':
            # self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
            #                             weight_decay=args.weight_decay)

            self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.weight_decay)
        else:
            raise Exception("optimizer not implement")

        # 定义学习率衰减
        if args.lr_scheduler == 'step':
            steps = [int(step) for step in args.steps.split(',')]
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=args.gamma)
        elif args.lr_scheduler == 'exp':
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.gamma)
        elif args.lr_scheduler == 'stepLR':
            steps = int(args.steps)
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, steps, args.gamma)
        elif args.lr_scheduler == 'fix':
            self.lr_scheduler = None
        else:
            raise Exception("lr schedule not implement")

        self.start_epoch = 0
        if args.resume:
            suffix = args.resume.rsplit('.', 1)[-1]
            if suffix == 'tar':
                checkpoint = torch.load(args.resume)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suffix == 'pth':
                self.model.load_state_dict(torch.load(args.resume, map_location=self.device))

        # Invert the model and define the loss
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        # #可视化输入特征
        # self.visualize_features_with_tsne()

        args = self.args

        step = 0
        best_acc = 0.0
        batch_count = 0
        batch_loss = 0.0
        batch_acc = 0
        step_start = time.time()

        save_list = Save_Tool(max_num=args.max_model_num)

        for epoch in range(self.start_epoch, args.max_epoch):

            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-' * 5)
            # Update the learning rate
            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))
            else:
                logging.info('current lr: {}'.format(args.lr))

            # Each epoch has a training and val phase
            for phase in ['train', 'val']:
                # Define the temp variable
                epoch_start = time.time()
                epoch_acc = 0
                epoch_loss = 0.0

                # Set model to train mode or test mode
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                sample_num = 0
                for data in self.dataloaders[phase]:
                    inputs = data.to(self.device)
                    labels = inputs.y
                    if args.task == 'Node':
                        bacth_num = inputs.num_nodes
                        sample_num += len(labels)
                    elif args.task == 'Graph':
                        bacth_num = inputs.num_graphs
                        sample_num += len(labels)
                    else:
                        print("There is no such task!!")
                    # Do the learning process, in val, we do not care about the gradient for relaxing
                    with torch.set_grad_enabled(phase == 'train'):

                        # forward
                        if args.task == 'Node':
                            logits = self.model(inputs)
                        elif args.task == 'Graph':
                            logits = self.model(inputs, args.pooltype)
                        else:
                            print("There is no such task!!")

                        loss = self.criterion(logits, labels)
                        pred = logits.argmax(dim=1)
                        correct = torch.eq(pred, labels).float().sum().item()
                        loss_temp = loss.item() * bacth_num
                        epoch_loss += loss_temp
                        epoch_acc += correct

                        # Calculate the training information
                        if phase == 'train':
                            # backward
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                            batch_loss += loss_temp
                            batch_acc += correct
                            batch_count += bacth_num

                            # Print the training information
                            if step % args.print_step == 0:
                                batch_loss = batch_loss / batch_count
                                batch_acc = batch_acc / batch_count
                                temp_time = time.time()
                                train_time = temp_time - step_start
                                step_start = temp_time
                                batch_time = train_time / args.print_step if step != 0 else train_time
                                sample_per_sec = 1.0 * batch_count / train_time
                                logging.info('Epoch: {}, Train Loss: {:.4f} Train Acc: {:.4f},'
                                             '{:.1f} examples/sec {:.2f} sec/batch'.format(
                                    epoch, batch_loss, batch_acc, sample_per_sec, batch_time
                                ))

                                batch_acc = 0
                                batch_loss = 0.0
                                batch_count = 0
                            step += 1

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                # Print the train and val information via each epoch

                epoch_loss = epoch_loss / sample_num
                epoch_acc = epoch_acc / sample_num

                logging.info('Epoch: {} {}-Loss: {:.4f} {}-Acc: {:.4f}, Cost {:.4f} sec'.format(
                    epoch, phase, epoch_loss, phase, epoch_acc, time.time() - epoch_start
                ))

                if phase == 'val':
                    # save the checkpoint for other learning
                    model_state_dic = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()
                    save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(epoch))
                    torch.save({
                        'epoch': epoch,
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'model_state_dict': model_state_dic
                    }, save_path)
                    save_list.update(save_path)
                    # save the best model according to the val accuracy
                    if epoch_acc > best_acc or epoch > args.max_epoch - 2:
                        best_acc = epoch_acc
                        logging.info("save best model epoch {}, acc {:.4f}".format(epoch, epoch_acc))
                        torch.save(model_state_dic,
                                   os.path.join(self.save_dir, '{}-{:.4f}-best_model.pth'.format(epoch, best_acc)))

                    # 在最后一个epoch或根据需要进行操作,混淆矩阵操作
                    if epoch == args.max_epoch - 1:
                        all_preds = []
                        all_labels = []
                        for data in self.dataloaders['val']:
                            # train
                        # for data in self.dataloaders['train']:
                            inputs = data.to(self.device)
                            labels = inputs.y
                            # 不计算梯度以加速和减少内存消耗
                            with torch.no_grad():
                                if args.task == 'Node':
                                    logits = self.model(inputs)
                                elif args.task == 'Graph':
                                    logits = self.model(inputs, args.pooltype)
                                preds = torch.argmax(logits, dim=1)
                            all_preds.extend(preds.cpu().numpy())
                            all_labels.extend(labels.cpu().numpy())

                        # 计算混淆矩阵
                        cm = confusion_matrix(all_labels, all_preds)
                        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化混淆矩阵
                        print("Confusion Matrix:")
                        print(cm_normalized)

                        # 绘制热力图
                        plt.figure(figsize=(10, 7))
                        sns.heatmap(cm_normalized, annot=True, fmt=".2%", cmap='Blues')
                        plt.xlabel('Predicted labels')
                        plt.ylabel('True labels')
                        plt.title('Confusion Matrix')
                        plt.show()
                    # 可视化
        self.visualize_features()

    def visualize_features(self):
        args = self.args

        features_list = []  # to store features
        labels_list = []  # to store labels

        # Ensure the model is in eval mode
        self.model.eval()

        # No need to track gradients for visualization
        with torch.no_grad():
            for inputs in self.dataloaders['val']:  # Assuming you want to visualize features from the validation set
                inputs = inputs.to(self.device)
                labels = inputs.y

                # Forward pass to get outputs
                if args.task == 'Node':
                    logits = self.model(inputs)
                elif args.task == 'Graph':
                    logits = self.model(inputs, args.pooltype)
                else:
                    print("There is no such task!!")

                # Collect features and labels
                features_list.append(logits.detach().cpu().numpy())
                labels_list.append(labels.cpu().numpy())

        # Convert collected features and labels to a suitable format for t-SNE
        features = np.concatenate(features_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)

        # Adjust perplexity if needed
        perplexity_value = min(30, len(features) // 3)  # Example adjustment, ensure perplexity < n_samples

        # Apply t-SNE to reduce dimensions to 3
        tsne = TSNE(n_components=3, verbose=1, perplexity=perplexity_value, n_iter=300)
        tsne_results = tsne.fit_transform(features)
        # Apply t-SNE to reduce dimensions to 2
        # tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity_value, n_iter=300)
        # tsne_results = tsne.fit_transform(features)

        # Visualization with matplotlib
        fig = plt.figure(figsize=(10, 7))
        # 三维
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2], c=labels,
                             cmap='hsv', alpha=0.5)
        # 二维
        # scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='hsv',
        #                       alpha=0.5)
        legend1 = ax.legend(*scatter.legend_elements(), title="类别")
        # plt.gca().add_artist(legend1)
        ax.add_artist(legend1)

        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Dimension 3')

        plt.show()

    def test(self):
        args = self.args
        # 加载数据集
        Dataset = getattr(datasets, args.data_name)
        self.datasets = {}
        self.datasets['val'] = Dataset(args.sample_length, args.data_dir, args.input_type,
                                                               args.task).data_prepare(test=True)
        """测试模型性能"""
        # 加载测试数据集
        test_dataloader = DataLoader(datasets, batch_size=self.args.batch_size, shuffle=False,
                                     num_workers=self.args.num_workers)

        # 加载训练好的模型
        model_path = os.path.join(args.data_dir, '108-0.9747-best_model.pth')
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)

        # 设置为评估模式
        self.model.eval()

        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for data in test_dataloader:
                inputs = data.to(self.device)
                labels = inputs.y  # 假设标签在输入数据的 `y` 属性
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()

        print(f'Test Accuracy: {100 * total_correct / total_samples:.2f}%')