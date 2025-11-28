import time
import os.path
import torch
import argparse
from pre_process import *
from utils import *
from exp import Experiments
import torch.multiprocessing as tmp
from utils import print_local_time, set_seed
import wandb

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["WANDB_MODE"] = "online"

parser = argparse.ArgumentParser()


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
parser.add_argument('--is_multi_parent', type=bool,
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
parser.add_argument('--cuda', type=bool, default=True,
                    help='use cuda for training')
parser.add_argument('--gpu_id', type=int, default=2, help='which gpu')
parser.add_argument('--seed', type=int, default=20,
                    help="seed for random generators")
parser.add_argument('--method', type=str, default='normal',
                    help='Experiment method conducted')

parser.add_argument('--exp_name', type=str,
                    default='my old story', help='Experiment name')
parser.add_argument('--resume', type=str, default='no', help='Resume a run')
parser.add_argument('--run_id', type=str, default='',
                    help='Wandb run id for resumption')
parser.add_argument('--entity', type=str, default='uaena', help='wandb entity')
parser.add_argument('--checkpoint_path', type=str,
                    default=None, help='checkpoint path to resume training')
parser.add_argument('--experiment_setting', type=str,
                    default='standard', help='experiment setting for SVGD')
parser.add_argument('--kernel_setting', type=str,
                    default='vmf_theta', help='kernel setting for SVGD')
parser.add_argument('--learn_mu', type=int, default=1,
                    help='If mu needs to be learnt during training')


start_time = time.time()
print("Start time at : ")
print_local_time()

args = parser.parse_args()


def experiment(args):
    torch.set_float32_matmul_precision('high')
    args.cuda = torch.cuda.is_available() and args.cuda == True

    if args.wandb == 1 and args.resume == 'no':
        wandb.init(
            project='PolarTaxo',

            name=f'{args.exp_name}-{args.dataset}',
            config=args,

        )

    if args.resume == 'must':
        run = wandb.init(
            entity=args.entity,
            project='PolarTaxo',
            id=args.run_id,
            resume='must'
        )

    if args.cuda:
        torch.cuda.set_device(args.gpu_id)

    print(args.cuda)

    print(args)

    set_seed(args.seed)
    if not os.path.isfile(os.path.join("../data/", args.dataset, "processed", "taxonomy_data_"+str(args.expID)+str(args.negsamples)+"_.pkl")):
        if args.dataset == 'computer_science' or args.dataset == 'psychology' or args.dataset == 'mesh' or args.dataset == 'wordnet_verb' or args.dataset == 'semeval_food':
            create_mag_data(args)
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
    print_local_time()
    print("************END***************")


# Additional Config
if __name__ == '__main__':
    args.gpu_id = 1
    args.expID = 14
    args.batch_size = 512
    args.accumulation_steps = 5
    args.geometric_weight = 0.7
    args.c = 0.4
    args.vmf_margin = 0.3
    args.svgd_weight = 0.1
    if args.dataset == 'mesh':
        args.negsamples = 20
        args.epochs = 50
        args.embed_size = 512
    elif args.dataset == 'wordnet_verb':
        args.negsamples = 20
        args.epochs = 30
        args.embed_size = 256
    experiment(args)
