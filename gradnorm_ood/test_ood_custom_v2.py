from utils import log
import resnetv2
import torch
import torchvision as tv
import time

import numpy as np

from utils.test_utils import arg_parser, get_measures
import os
import glob

from sklearn.linear_model import LogisticRegressionCV
from torch.autograd import Variable
from utils.mahalanobis_lib import get_Mahalanobis_score
import resnet101
import random
from collections import Counter
import matplotlib.pyplot as plt


class IDDataset(torch.utils.data.Dataset):
    def __init__(self, path, meta_file, transform, loader=tv.datasets.folder.default_loader):
        super().__init__()
        file_list = []
        label_list = []
        with open(meta_file,'r') as f:
            ls = f.readlines()
            # random.shuffle(ls)
            for l in ls:
                name, label = l.strip().split(' ')
                file_list.append(os.path.join(path,name))
                label_list.append(label)
        self.file_list = file_list
        self.label_list = label_list
        self.loader = loader
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        path = self.file_list[item]
        sample = self.loader(path)
        label = self.label_list[item]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label



def make_id_ood(args, logger):
    """Returns train and validation datasets."""
    crop = 480

    val_tx = tv.transforms.Compose([
        tv.transforms.Resize((crop, crop)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([123.675/255, 116.28/255, 103.53/255],
                                [58.395/255, 57.12/255, 57.375/255]),
    ])

    # in_set = tv.datasets.ImageFolder(args.in_datadir, val_tx)
    in_set = IDDataset(args.in_datadir, args.meta_file, val_tx)
    out_set = tv.datasets.ImageFolder(args.out_datadir, val_tx)

    logger.info(f"Using an in-distribution set with {len(in_set)} images.")
    logger.info(f"Using an out-of-distribution set with {len(out_set)} images.")

    in_loader = torch.utils.data.DataLoader(
        in_set, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    out_loader = torch.utils.data.DataLoader(
        out_set, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    return in_set, out_set, in_loader, out_loader


def iterate_data_msp(data_loader, model):
    confs = []
    m = torch.nn.Softmax(dim=-1).cuda()
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            # compute output, measure accuracy and record loss.
            logits = model(x)

            conf, _ = torch.max(m(logits), dim=-1)
            confs.extend(conf.data.cpu().numpy())
    return np.array(confs)


def iterate_data_odin(data_loader, model, epsilon, temper, logger):
    criterion = torch.nn.CrossEntropyLoss().cuda()
    confs = []
    for b, (x, y) in enumerate(data_loader):
        x = Variable(x.cuda(), requires_grad=True)
        outputs = model(x)

        maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)
        outputs = outputs / temper

        labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
        loss = criterion(outputs, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(x.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        # Adding small perturbations to images
        tempInputs = torch.add(x.data, -epsilon, gradient)
        outputs = model(Variable(tempInputs))
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

        confs.extend(np.max(nnOutputs, axis=1))
        if b % 100 == 0:
            logger.info('{} batches processed'.format(b))


    return np.array(confs)


def iterate_data_energy(data_loader, model, temper):
    confs = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            # compute output, measure accuracy and record loss.
            logits = model(x)

            conf = temper * torch.logsumexp(logits / temper, dim=1)
            confs.extend(conf.data.cpu().numpy())
    return np.array(confs)


def iterate_data_mahalanobis(data_loader, model, num_classes, sample_mean, precision,
                             num_output, magnitude, regressor, logger):
    confs = []
    for b, (x, y) in enumerate(data_loader):
        if b % 10 == 0:
            logger.info('{} batches processed'.format(b))
        x = x.cuda()

        Mahalanobis_scores = get_Mahalanobis_score(x, model, num_classes, sample_mean, precision, num_output, magnitude)
        scores = -regressor.predict_proba(Mahalanobis_scores)[:, 1]
        confs.extend(scores)
    return np.array(confs)


def iterate_data_gradnorm(data_loader, model, temperature, num_classes):
    confs = []
    cls = []
    with torch.no_grad():
        for b, (x, y) in enumerate(data_loader):
            if b % 10 == 0:
                print('{} batches processed'.format(b))
            inputs = Variable(x.cuda(), requires_grad=False)
            outputs, features = model(inputs)
            U = torch.norm(features, p=1, dim=1)
            out_softmax = torch.nn.functional.softmax(outputs, dim=1)
            V = torch.norm((1 - 1000 * out_softmax), p=1, dim=1)
            S = U * V / 2048 / 1000
            S = S.cpu().numpy().tolist()
            confs.extend(S)
            cls.extend(y)
            # print(S)
            # # debug
            # if b > 50:
            #    break
    for i in range(len(cls)):
        item = cls[i]
        cls[i] = int(item)
    return np.array(confs), np.array(cls)

def iterate_data_gradnorm_custom(data_loader, model, temperature, num_classes, id_cls=True):
    confs = []
    cls = []
    logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()
    with torch.no_grad():
        for b, (x, y) in enumerate(data_loader):
            if b % 10 == 0:
                print('{} batches processed'.format(b))
            inputs = Variable(x.cuda(), requires_grad=False)
            outputs, features = model(inputs)
            targets = torch.ones((inputs.shape[0], num_classes)).cuda()
            outputs = outputs / temperature
            loss = torch.sum(-targets * logsoftmax(outputs), dim=-1)
            # loss = torch.mean(a,dim=-1)
            loss = loss.detach().cpu().numpy().tolist()
            confs.extend(loss)
            cls.extend(y)
    return np.array(confs), np.array(cls)



def run_eval(model, in_loader, out_loader, logger, args, num_classes):
    # switch to evaluate mode
    model.eval()

    logger.info("Running test...")
    logger.flush()

    if args.score == 'MSP':
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_msp(in_loader, model)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_msp(out_loader, model)
    elif args.score == 'ODIN':
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_odin(in_loader, model, args.epsilon_odin, args.temperature_odin, logger)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_odin(out_loader, model, args.epsilon_odin, args.temperature_odin, logger)
    elif args.score == 'Energy':
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_energy(in_loader, model, args.temperature_energy)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_energy(out_loader, model, args.temperature_energy)
    elif args.score == 'Mahalanobis':
        sample_mean, precision, lr_weights, lr_bias, magnitude = np.load(
            os.path.join(args.mahalanobis_param_path, 'results.npy'), allow_pickle=True)
        sample_mean = [s.cuda() for s in sample_mean]
        precision = [p.cuda() for p in precision]

        regressor = LogisticRegressionCV(cv=2).fit([[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]],
                                                   [0, 0, 1, 1])

        regressor.coef_ = lr_weights
        regressor.intercept_ = lr_bias

        temp_x = torch.rand(2, 3, 480, 480)
        temp_x = Variable(temp_x).cuda()
        temp_list = model(x=temp_x, layer_index='all')[1]
        num_output = len(temp_list)

        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_mahalanobis(in_loader, model, num_classes, sample_mean, precision,
                                             num_output, magnitude, regressor, logger)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_mahalanobis(out_loader, model, num_classes, sample_mean, precision,
                                              num_output, magnitude, regressor, logger)
    elif args.score == 'GradNorm':
        logger.info("Processing in-distribution data...")
        in_scores, id_labels = iterate_data_gradnorm(in_loader, model, args.temperature_gradnorm, num_classes)
        # logger.info("Processing out-of-distribution data...")
        out_scores, _ = iterate_data_gradnorm(out_loader, model, args.temperature_gradnorm, num_classes)
    else:
        raise ValueError("Unknown score type {}".format(args.score))
    norm = 1
    in_examples = in_scores.reshape((-1, 1)) / norm
    out_examples = out_scores.reshape((-1, 1)) / norm
    # id_labels = id_labels.reshape((-1, 1))

    id_cls = []
    cls_idx = []
    label_filename = args.id_cls
    a = label_filename[-5]
    with open(label_filename, 'r') as f:
        for line in f.readlines():
            segs = line.strip().split(' ')
            cls_idx.append(int(segs[-1]))
    cls_idx = np.array(cls_idx, dtype='int')
    res = Counter(cls_idx)
    for item in id_labels:
        id_cls.append(res[item])
    id_cls = np.array(id_cls).reshape((-1, 1))
    nums = np.random.randint(0, 50000, 10000)
    in_examples_ = in_examples[nums]
    id_cls_ = id_cls[nums]
    plt.figure(1)
    plt.scatter(in_examples_, id_cls_)
    plt.savefig('./files/test_{}_gn.jpg'.format(a))

    #generate feature files
    # with open('files/LT_a8_in_score.txt','w') as f:
    #     for item in in_examples:
    #         f.write('{}\n'.format(item))
    # with open('files/LT_a8_out_score.txt','w') as f:
    #     for item in out_examples:
    #         f.write('{}\n'.format(item))
    # import time
    # with open('files/LT_id_label_{}.txt'.format(time.time()),'w') as f:
    #     for item in id_cls:
    #         f.write('{}\n'.format(item))
    # exit()
    #overall eval
    auroc, aupr_in, aupr_out, fpr95 = get_measures(in_examples, out_examples)

    #head-tail eval
    head_examples = []
    mid_examples = []
    tail_examples = []
    head_cls = []
    mid_cls = []
    tail_cls = []
    head_ratio = 0
    mid_ratio = 0
    tail_ratio = 0
    thr = 0.35
    for i in range(id_cls.shape[0]):
        if id_cls[i] >= 100:
            head_examples.append(in_examples[i])
            head_cls.append(id_cls[i])
        elif 20 < id_cls[i] < 100:
            mid_examples.append(in_examples[i])
            mid_cls.append(id_cls[i])
        elif 0 <= id_cls[i] <= 20:
            tail_examples.append(in_examples[i])
            tail_cls.append(id_cls[i])
        else:
            raise ValueError("Unknown id class type {}".format(id_cls[i]))
    out_examples = out_examples.tolist()
    random.shuffle(out_examples)
    auroc_head, aupr_in_head, aupr_out_head, fpr95_head = [0,0,0,0]
    if len(head_examples) > 0:
        head_examples = np.stack(head_examples,axis=0)
        head_ratio = int(head_examples.shape[0] / in_examples.shape[0] * len(out_examples))
        head_out_examples = np.array(out_examples[0:head_ratio])
        plt.figure(2)
        plt.scatter(head_examples, head_cls)
        plt.savefig('./files/test_head_{}_gn.jpg'.format(a))
        auroc_head, aupr_in_head, aupr_out_head, fpr95_head = get_measures(head_examples, head_out_examples)

    auroc_mid, aupr_in_mid, aupr_out_mid, fpr95_mid = [0,0,0,0]
    if len(mid_examples) > 0:
        mid_examples = np.stack(mid_examples,axis=0)
        mid_ratio = int(mid_examples.shape[0] / in_examples.shape[0] * len(out_examples))
        mid_out_examples = np.array(out_examples[0:mid_ratio])
        plt.figure(3)
        plt.scatter(mid_examples, mid_cls)
        plt.savefig('./files/test_mid_{}_gn.jpg'.format(a))
        auroc_mid, aupr_in_mid, aupr_out_mid, fpr95_mid = get_measures(mid_examples, mid_out_examples)

    auroc_tail, aupr_in_tail, aupr_out_tail, fpr95_tail=[0,0,0,0]
    if len(tail_examples) > 0:
        tail_examples = np.stack(tail_examples,axis=0)
        tail_ratio = int(tail_examples.shape[0] / in_examples.shape[0] * len(out_examples))
        tail_out_examples = np.array(out_examples[0:tail_ratio])
        plt.figure(4)
        plt.scatter(tail_examples, tail_cls)
        plt.savefig('./files/test_tail_{}_gn.jpg'.format(a))
        auroc_tail, aupr_in_tail, aupr_out_tail, fpr95_tail = get_measures(tail_examples, tail_out_examples)
    in_examples_ = in_examples > thr
    overall_acc = np.sum(in_examples_) / len(in_examples_)

    logger.info('============Overall Results for {}============'.format(args.score))
    logger.info('AUROC: {}'.format(auroc))
    logger.info('AUPR (In): {}'.format(aupr_in))
    logger.info('AUPR (Out): {}'.format(aupr_out))
    logger.info('FPR95: {}'.format(fpr95))
    logger.info('quick data:{},{},{},{}'.format(auroc, aupr_in, aupr_out, fpr95))
    logger.info('============Head Results for {}============'.format(args.score))
    logger.info('AUROC: {}'.format(auroc_head))
    logger.info('AUPR (In): {}'.format(aupr_in_head))
    logger.info('AUPR (Out): {}'.format(aupr_out_head))
    logger.info('FPR95: {}'.format(fpr95_head))
    logger.info('quick data:{},{},{},{}'.format(auroc_head, aupr_in_head, aupr_out_head, fpr95_head))
    logger.info('============Mid Results for {}============'.format(args.score))
    logger.info('AUROC: {}'.format(auroc_mid))
    logger.info('AUPR (In): {}'.format(aupr_in_mid))
    logger.info('AUPR (Out): {}'.format(aupr_out_mid))
    logger.info('FPR95: {}'.format(fpr95_mid))
    logger.info('quick data:{},{},{},{}'.format(auroc_mid, aupr_in_mid, aupr_out_mid, fpr95_mid))
    logger.info('============Tail Results for {}============'.format(args.score))
    logger.info('AUROC: {}'.format(auroc_tail))
    logger.info('AUPR (In): {}'.format(aupr_in_tail))
    logger.info('AUPR (Out): {}'.format(aupr_out_tail))
    logger.info('FPR95: {}'.format(fpr95_tail))
    logger.info('quick data:{},{},{},{}'.format(auroc_tail, aupr_in_tail, aupr_out_tail, fpr95_tail))
    logger.flush()

def main(args):
    logger = log.setup_logger(args)

    torch.backends.cudnn.benchmark = True

    # if args.score == 'GradNorm':
    #     args.batch = 1

    in_set, out_set, in_loader, out_loader = make_id_ood(args, logger)

    logger.info(f"Loading model from {args.model_path}")

    # model = resnetv2.KNOWN_MODELS[args.model](head_size=1000)
    # state_dict = torch.load(args.model_path)
    # model.load_state_dict_custom(state_dict['model'])
    model = resnet101.resnet101()
    b = dict()
    a = torch.load(args.model_path)
    for k, v in a["state_dict"].items():
        b[".".join(k.split(".")[1:])] = v
    model.load_state_dict(b)



    if args.score != 'GradNorm':
        model = torch.nn.DataParallel(model)

    model = model.eval().cuda()

    start_time = time.time()
    run_eval(model, in_loader, out_loader, logger, args, num_classes=1000)
    end_time = time.time()

    logger.info("Total running time: {}".format(end_time - start_time))


if __name__ == "__main__":
    parser = arg_parser()

    parser.add_argument("--in_datadir", help="Path to the in-distribution data folder.")
    parser.add_argument("--out_datadir", help="Path to the out-of-distribution data folder.")
    parser.add_argument("--meta_file", default='',help="Path to the in-of-distribution data txt.")
    parser.add_argument("--id_cls", default='', help="id label file")


    parser.add_argument('--score', choices=['MSP', 'ODIN', 'Energy', 'Mahalanobis', 'GradNorm'], default='GradNorm')

    # arguments for ODIN
    parser.add_argument('--temperature_odin', default=1000, type=int,
                        help='temperature scaling for odin')
    parser.add_argument('--epsilon_odin', default=0.0, type=float,
                        help='perturbation magnitude for odin')

    # arguments for Energy
    parser.add_argument('--temperature_energy', default=1, type=int,
                        help='temperature scaling for energy')

    # arguments for Mahalanobis
    parser.add_argument('--mahalanobis_param_path', default='checkpoints/finetune/tune_mahalanobis',
                        help='path to tuned mahalanobis parameters')

    # arguments for GradNorm
    parser.add_argument('--temperature_gradnorm', default=1, type=int,
                        help='temperature scaling for GradNorm')

    main(parser.parse_args())
