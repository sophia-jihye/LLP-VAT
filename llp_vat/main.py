import argparse
import os
import pickle
import uuid

import arrow
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from tqdm.auto import tqdm

from llp_vat.lib.llp import (BagMiniBatch, BagSampler, Iteration,
                             load_llp_dataset)
from llp_vat.lib.losses import (PiModelLoss, ProportionLoss, VATLoss,
                                compute_hard_l1, compute_soft_kl)
from llp_vat.lib.models import MLPModel
from llp_vat.lib.ramps import sigmoid_rampup
from llp_vat.lib.run_experiment import (RunExperiment, save_checkpoint,
                                        write_meters)
from llp_vat.lib.utils import AverageMeterSet, accuracy, parameters_string


def train_valid_split(dataset, valid_ratio, seed):
    torch.manual_seed(seed)
    valid_size = int(valid_ratio * len(dataset))
    train_size = len(dataset) - valid_size
    train, valid = random_split(dataset, [train_size, valid_size])
    return train, valid


def get_rampup_weight(weight, iteration, rampup):
    alpha = weight * sigmoid_rampup(iteration.value, rampup)
    return alpha


def train_llp(args, epoch, iteration, model, optimizer, loader,
              criterion, consistency_criterion, logger):
    meters = AverageMeterSet()

    mini_batch = BagMiniBatch(args.n_samples)
    # set up training mode for model
    model.train()
    for i, (x, y) in tqdm(enumerate(loader),
                          "[train#{}]".format(epoch),
                          leave=False,
                          ncols=150,
                          total=len(loader),
                          disable=args.disable):
        with torch.autograd.set_detect_anomaly(True):
            x = x.cuda()
            y = y.cuda()

            # accumulate x until the batch size is greater than or equal to
            # the buffer size
            mini_batch.append(x, y)
            if mini_batch.num_bags < args.mini_batch_size:
                continue

            # skip training if there exists only one instance in a mini-batch
            # because the BatchNorm would crash under this circumstance
            if mini_batch.total_size == 1:
                continue

            # concatenate all bags
            x, y = map(torch.cat, zip(*mini_batch.bags))

            logits = None
            if args.consistency_type == "vat":
                # VAT should be calculated before the forward for cross entropy
                consistency_loss = consistency_criterion(model, x)
            elif args.consistency_type == "pi":
                consistency_loss, logits = consistency_criterion(model, x)
            else:
                consistency_loss = torch.tensor(0.)
            alpha = get_rampup_weight(args.consistency, iteration,
                                      args.consistency_rampup)
            consistency_loss = alpha * consistency_loss
            meters.update("cons_loss", consistency_loss.item())
            meters.update("cons_weight", alpha)

            # reuse the logits from pi-model
            if logits is None:
                logits = model(x)
            probs = F.softmax(logits, dim=1)

            # compute proportion loss for each bag
            if args.alg == "uniform":
                # compute propotion loss in the batch way
                batch_probs = probs.view(
                    mini_batch.num_bags, args.bag_size, -1)
                batch_avg_probs = torch.mean(batch_probs, dim=1)
                batch_target = torch.stack(mini_batch.targets)
                prop_loss = criterion(batch_avg_probs, batch_target)
            else:
                # compute proportion loss in sequential way
                prop_loss = 0
                start = 0
                for bag_size, target in mini_batch:
                    # proportion loss
                    avg_probs = torch.mean(
                        probs[start:start + bag_size], dim=0)
                    prop_loss += criterion(avg_probs, target)
                    start += bag_size
                prop_loss = prop_loss / mini_batch.num_bags
            meters.update("prop_loss", prop_loss.item())

            # proportion_loss + consistency_loss
            loss = prop_loss + consistency_loss
            meters.update("loss", loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iteration.step()

            prec1, prec5 = accuracy(logits, y.argmax(1), top_k=(1, 5))
            meters.update("top1", prec1.item(), y.size(0))
            meters.update("top5", prec5.item(), y.size(0))

            # clear mini_batch
            mini_batch.reset()
    if logger:
        logger.info("Epoch#{}-{} "
                    "cons_weight={meters[cons_weight].avg:.4f} "
                    "cons_loss={meters[cons_loss].avg:.4f} "
                    "prop_loss={meters[prop_loss].avg:.4f} "
                    "loss={meters[loss].avg:.4f} "
                    "prec@1={meters[top1].avg:.2f}% "
                    "prec@5={meters[top5].avg:.2f}%".format(epoch,
                                                            iteration.value,
                                                            meters=meters))
    return meters


def eval(args, num_classes, epoch, iteration, model, loader, criterion, logger, prefix=""):
    meters = AverageMeterSet()

    model.eval()
    for x, y in tqdm(loader,
                     "[Evalutaion]",
                     leave=False,
                     ncols=150,
                     disable=args.disable):
        x = x.cuda()
        y = y.cuda()

        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            avg_probs = torch.mean(probs, dim=0)
            avg_ys = torch.mean(y, dim=0)
            soft_kl = compute_soft_kl(avg_probs, avg_ys)
            hard_l1 = compute_hard_l1(probs, y, num_classes)
            loss = criterion(avg_probs, avg_ys)
        meters.update('soft_kl', soft_kl.item())
        meters.update('hard_l1', hard_l1.item())
        meters.update('prop_loss', loss.item())

        prec1, prec5 = accuracy(logits, y.argmax(1), top_k=(1, 5))
        meters.update('top1', prec1.item(), y.size(0))
        meters.update('top5', prec5.item(), y.size(0))
    if logger:
        logger.info("Epoch#{}-{} "
                    "{prefix}soft_kl={meters[soft_kl].avg:.4f} "
                    "{prefix}hard_l1={meters[hard_l1].avg:.4f} "
                    "{prefix}prop_loss={meters[prop_loss].avg:.4f} "
                    "{prefix}prec@1={meters[top1].avg:.2f}% "
                    "{prefix}prec@5={meters[top5].avg:.2f}%".format(
                        epoch, iteration.value, meters=meters, prefix=prefix))
    return meters


def inference(model, loader):
    records = []

    model.eval()
    for x, y in tqdm(loader,
                     "[Inference]",
                     leave=False,
                     ncols=150,
                     disable=args.disable):
        x = x.cuda()

        with torch.no_grad():
            logits = model(x)
            records.append((x.squeeze(), logits.argmax().item(), y.argmax().item()))            
    return records

def run_experiment(args, experiment):
    experiment.save_config(vars(args))

    # create logger for training, testing, validation
    logger = experiment.create_logfile("experiment")
    train_log = experiment.create_logfile("train")
    valid_log = experiment.create_logfile("valid")
    test_log = experiment.create_logfile("test")
    # create tensorboard writer
    tb_writer = experiment.create_tb_writer()
    logger.info(args)

    # load LLP dataset
    if args.alg == "uniform":
        dataset, bags = load_llp_dataset(args.domain_index,
                                         args.obj_dir,
                                         args.alg,
                                         replacement=args.replacement,
                                         bag_size=args.bag_size)
    elif args.alg == "kmeans":
        dataset, bags = load_llp_dataset(args.domain_index,
                                         args.obj_dir,
                                         args.alg,
                                         n_clusters=args.n_clusters,
                                         reduction=args.reduction)
    else:
        raise NameError("The bag creation algorithm is unknown")

    # consturct data loader
    train_bags, valid_bags = train_valid_split(bags, args.valid, args.seed)
    train_bag_sampler = BagSampler(train_bags, args.num_bags)
    train_loader = DataLoader(dataset["train"],
                              batch_sampler=train_bag_sampler,
                              pin_memory=True,
                              num_workers=2)
    valid_loader = None
    if args.valid > 0:
        valid_bag_sampler = BagSampler(valid_bags, num_bags=-1)
        valid_loader = DataLoader(dataset["train"],
                                  batch_sampler=valid_bag_sampler,
                                  pin_memory=True,
                                  num_workers=2)
    test_loader = DataLoader(dataset["test"],
                             batch_size=256,
                             pin_memory=True,
                             num_workers=2)

    # declare model
    model = MLPModel(dataset["input_dim"], dataset["num_classes"])
    model = model.cuda()

    # declare optimizer
    if args.optimizer.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(),
                              momentum=0.9,
                              lr=args.lr,
                              weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "adam":
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr,
                               weight_decay=args.weight_decay)
    else:
        raise NameError("optimizer {} is not supported".format(args.optimizer))

    # print model architecture and optimizer
    logger.info(parameters_string(model))
    logger.info(optimizer)

    # declare LLP criterion - the Proportion loss
    criterion = ProportionLoss(args.metric, 1.0)
    logger.info(criterion)

    # declare consistency criterion
    if args.consistency_type == "none":
        consistency_criterion = None
    elif args.consistency_type == "vat":
        consistency_criterion = VATLoss(xi=args.xi, eps=args.eps, ip=args.ip)
    elif args.consistency_type == "pi":
        consistency_criterion = PiModelLoss(std=args.std)
    else:
        raise NameError("Unknown consistency criterion")

    if consistency_criterion and args.consistency_rampup == -1:
        args.consistency_rampup = 0.4 * args.num_epochs * \
            len(train_loader) / args.mini_batch_size

    # ajust learning rate
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=320, gamma=0.2)

    iteration = Iteration()
    for epoch in range(args.num_epochs):
        train_meters = train_llp(args, epoch, iteration, model,
                                 optimizer, train_loader, criterion,
                                 consistency_criterion, train_log)
        write_meters(epoch, "train", tb_writer, train_meters)

        if valid_loader:
            valid_meters = eval(args, dataset["num_classes"], epoch, iteration, model, valid_loader,
                                criterion, valid_log)
            write_meters(epoch, "valid", tb_writer, valid_meters)

        test_meters = eval(args, dataset["num_classes"], epoch, iteration, model, test_loader,
                           criterion, test_log)
        write_meters(epoch, "test", tb_writer, test_meters)

        scheduler.step()

    # save checkpoint
    logger.info("Save checkpoint#{}".format(epoch))
    filename = os.path.join(experiment.result_dir, "model.tar")
    save_checkpoint(filename, model, epoch, optimizer)

    tb_writer.close()

    # Inference
    inference_loader = DataLoader(dataset["test"],
                             batch_size=1,
                             pin_memory=True,
                             num_workers=2)
    records = inference(model, inference_loader)
    logger.info("Save weak labels for GSAD domain {}".format(args.domain_index))
    filename = os.path.join(experiment.result_dir, "weak_labels_GSAD{}.pkl".format(args.domain_index))
    with open(filename, 'wb') as f:
        pickle.dump(records, f)
    print('Created', filename)

def main(args):
    uid = "{time}_{uuid}".format(
        time=arrow.utcnow().format("YYYYMMDDTHH:mm:ss"),
        uuid=str(uuid.uuid4())[:4]
    )
    result_dir = os.path.join(args.result_dir, uid)
    experiment = RunExperiment(result_dir)
    run_experiment(args, experiment)


def get_args():
    parser = argparse.ArgumentParser(
        "Learning from Label Proportions with Consistency Regularization")

    # basic arguments
    parser.add_argument("--obj_dir", default="./obj")
    parser.add_argument("-i", "--domain_index", type=int, default=10)
    parser.add_argument("--result_dir", default="./results")
    parser.add_argument("-e", "--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--weight_decay", type=float, default=0.)
    parser.add_argument("--metric", type=str, default="ce")
    parser.add_argument("--valid", type=float, default=0.1)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--n_samples", default=0, type=int)
    parser.add_argument("--disable", action="store_true",
                        help="disable the progress bar")

    # bag creation algorithms
    parser.add_argument("--alg", choices=["uniform", "kmeans"], default='uniform')
    parser.add_argument("-b", "--bag_size", type=int, default=64)
    parser.add_argument("--replacement", action="store_true")
    parser.add_argument("-k", "--n_clusters", type=int)
    parser.add_argument("--reduction", type=int, default=600)

    # coefficient for proportion loss
    parser.add_argument("--num_bags", default=-1, type=int)
    parser.add_argument("--mini_batch_size", type=int, default=2)

    # consistency args
    parser.add_argument("--consistency_type",
                        choices=["vat", "pi", "none"],
                        default="vat")
    parser.add_argument("--consistency", type=float, default=0.05)
    parser.add_argument("--consistency_rampup", type=int, default=-1)
    # pi args
    parser.add_argument("--std", type=float, default=0.15)
    # vat args
    parser.add_argument("--xi", type=float, default=1e-6)
    parser.add_argument("--eps", type=float, default=6.0)
    parser.add_argument("--ip", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    main(args)
