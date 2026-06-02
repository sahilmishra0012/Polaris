import time
import os.path
import torch
import argparse
from pre_process import *
from utils import *
from exp import Experiments
import torch.multiprocessing as tmp

import wandb

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["WANDB_MODE"] = "online"

parser = argparse.ArgumentParser()


def str2bool(value):
    """Parse common command-line boolean spellings."""
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


# General ML hyperparameters
parser.add_argument('--dataset', type=str,
                    default='environment', help='dataset')
parser.add_argument('--pre_train', type=str,
                    default="bert", help='Pre_trained model')
parser.add_argument('--hidden', type=int, default=64,
                    help='dimension of hidden layers in MLP')
parser.add_argument('--dropout', type=float, default=0.4, help='dropout')
parser.add_argument('--wandb', type=int, default=1,
                    help="Enable wandb logging")
parser.add_argument('--mixture', type=str, default=None,
                    help="Type of weighting in mixture model")
parser.add_argument('--padmaxlen', type=int, default=30,
                    help='max length of padding')
parser.add_argument('--matrixsize', type=int, default=768,
                    help="Size of density matrix")
parser.add_argument('--negsamples', type=int, default=20,
                    help="Number of negative samples per node")
parser.add_argument('--model', type=str, default='bert',
                    help='Pretained Language Model')

parser.add_argument('--expID', type=int, default=8, help='-th of experiments')
parser.add_argument('--epochs', type=int, default=55, help='training epochs')
parser.add_argument('--batch_size', type=int, default=512,
                    help='training batch size')
parser.add_argument('--lr', type=float, default=9e-5,
                    help='learning rate for pre-trained model')
parser.add_argument('--lr_proj', type=float, default=1e-3,
                    help='learning rate for pre-trained model')
parser.add_argument('--eps', type=float, default=1e-8, help='adamw_epsilon')
parser.add_argument('--optim', type=str, default="adamw", help='Optimizer')
parser.add_argument('--embed_size', type=int, default=8, help='Embedding Size')
parser.add_argument('--accumulation_steps', type=int, default=5,
                    help='Increase accumulation steps to use Gradient Accumulation')


# Polar related
parser.add_argument('--beta', type=float, default=0.5,
                    help='Negative sample margin')
parser.add_argument('--vmf_margin', type=float,
                    default=0.5, help='Margin for VMF loss')
parser.add_argument('--c', type=float, default=0.7,
                    help='parameter c for welsch loss')
parser.add_argument('--is_multi_parent', type=str2bool,
                    default=True, help='If it is a multi parent taxonomy')
parser.add_argument('--geometric_weight', type=float,
                    default=0.5, help='Importance of Geometric Loss')
parser.add_argument('--probabilistic_weight', type=float,
                    default=0.5, help='Importance of Probabilistic Loss')
parser.add_argument(
    '--svgd_weight', help='Importance of Svgd Loss', type=float)
parser.add_argument('--kappa_align', help='Kappa Alignment',
                    type=float, default=2.5)
parser.add_argument('--kappa_repel', help='Kappa repulsion',
                    type=float, default=4.5)

# Others
parser.add_argument('--cuda', type=str2bool, default=True,
                    help='use cuda for training')
parser.add_argument('--gpu_id', type=int, default=2, help='which gpu')
parser.add_argument('--seed', type=int, default=20,
                    help="seed for random generators")
parser.add_argument('--method', type=str, default='normal',
                    help='Experiment method conducted')

parser.add_argument('--exp_name', type=str,
                    default='experiment_name', help='Experiment name')
parser.add_argument('--resume', type=str, default='no', help='Resume a run')
parser.add_argument('--run_id', type=str, default='',
                    help='Wandb run id for resumption')
parser.add_argument('--entity', type=str,
                    default='uaena', help='wandb entity')
parser.add_argument('--checkpoint_path', type=str,
                    default=None, help='checkpoint path to resume training')
parser.add_argument('--experiment_setting', type=str,
                    default='standard', help='experiment setting for SVGD')
parser.add_argument('--kernel_setting', type=str,
                    default='vmf_theta', help='kernel setting for SVGD')
parser.add_argument('--learn_mu', type=int, default=1,
                    help='If mu needs to be learned during training')
parser.add_argument('--detach_svgd', type=int, default=0,
                    help='detach SVGD from training')
parser.add_argument('--learn_kappa', type=int, default=1,
                    help='If kappa parameter of VMF needs to be learned during training')
parser.add_argument('--implement_rectangular_opt', type=str2bool, default=False,
                    help='Optimizes parameters on a Grid instead of a Sphere')
parser.add_argument('--potential_strength', type=float,
                    default=0.5, help='Potential Strength')


start_time = time.time()
print("Start time at : ")
print_aoe_time()

args = parser.parse_args()


def experiment(args):
    """Run preprocessing when needed, initialize the experiment driver, and train the model."""
    torch.set_float32_matmul_precision('high')
    args.cuda = torch.cuda.is_available() and args.cuda == True

    if args.wandb == 1 and args.resume == 'no':
        wandb.init(
            project='Polaris',

            name=f'{args.exp_name}-{args.dataset}',
            config=args,

        )

    if args.resume == 'must':
        run = wandb.init(
            entity=args.entity,
            project='Polaris',
            id=args.run_id,
            resume='must'
        )

    if args.cuda:
        torch.cuda.set_device(args.gpu_id)

    print(args.cuda)

    print(args)

    set_seed(args.seed)
    if not os.path.isfile(os.path.join("../data/", args.dataset, "processed", "taxonomy_data_"+str(args.expID)+str(args.negsamples)+str(args.seed)+"_.pkl")):
        if args.dataset == 'computer_science' or args.dataset == 'psychology' or args.dataset == 'mesh' or args.dataset == 'wordnet_verb' or args.dataset == 'semeval_food':
            create_mag_data(args)
        elif args.dataset == 'birds':
            create_image_data(args)
        else:
            create_data(args)

    # args.expID = wandb.run

    exp = Experiments(args)
    exp.train(checkpoint=args.checkpoint_path)
    """Train the model"""

    # exp.predict(tag="test")
    # exp.save_prediction()

    wandb.finish()

    print("Time used :{:.01f}s".format(time.time()-start_time))
    print("End time at : ")

    print("************END***************")


if __name__ == '__main__':
    args.gpu_id = 0
    args.expID = 510
    args.batch_size = 1024
    args.accumulation_steps = 5
    args.geometric_weight = 0.7
    args.c = 0.4
    args.vmf_margin = 0.3
    args.svgd_weight = 0.1
    if args.dataset == 'mesh':
        args.negsamples = 20
        args.epochs = 50
        args.embed_size = 128
        args.batch_size = 256
    elif args.dataset == 'wordnet_verb':
        args.negsamples = 20
        args.epochs = 30
        args.embed_size = 256
    elif args.dataset == 'birds':
        args.negsamples = 5
        args.epochs = 150
        args.embed_size = 256
    elif args.dataset == 'semeval_food':
        args.gpu_id = 1
        args.epochs = 200
        args.negsamples = 20
        args.geometric_weight = 0.7
        args.embed_size = 64
        args.beta = 0.3
        args.kappa_align = 1.0
        args.kappa_repel = 2.0
    elif args.dataset == 'science' or args.dataset == 'environment' or args.dataset == 'wordnet':
        args.is_multi_parent = False
        args.epochs = 90
        args.accumulation_steps = 2
        args.negsamples = 50
        args.kappa_align = 1.0
        args.kappa_repel = 2.0
        if args.dataset == 'wordnet':
            args.batch_size = 64
            args.kappa_align = 2
            args.kappa_repel = 4
            args.embed_size = 128
            args.negsamples = 15
            args.epochs = 100
            args.gpu_id = 2
        elif args.dataset == 'science':
            args.embed_size = 64
            args.batch_size = 128
            args.epochs = 50
            args.gpu_id = 1
            args.beta = 0.3
        elif args.dataset == 'environment':
            args.embed_size = 32
            args.batch_size = 128
            args.gpu_id = 0
            args.epochs = 50
            args.beta = 0.3

    experiment(args)
