import time
import torch
import argparse
from pre_process import *
from utils import *
from exp import Experiments
from utils import print_local_time, set_seed
import os


parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str,
                    default='environment', help='dataset')
parser.add_argument('--pre_train', type=str,
                    default="bert", help='Pre_trained model')
parser.add_argument('--hidden', type=int, default=64,
                    help='dimension of hidden layers in MLP')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--wandb', type=int, default=0,
                    help="Enable wandb logging")
parser.add_argument('--padmaxlen', type=int, default=128,
                    help='max length of padding')
parser.add_argument('--matrixsize', type=int, default=768,
                    help="Size of density matrix")
parser.add_argument('--negsamples', type=int, default=50,
                    help="Number of negative samples per node")

# Training hyper-parameters
parser.add_argument('--expID', type=int, default=0, help='-th of experiments')
parser.add_argument('--epochs', type=int, default=100, help='training epochs')
parser.add_argument('--batch_size', type=int, default=128,
                    help='training batch size')
parser.add_argument('--lr', type=float, default=2e-5,
                    help='learning rate for pre-trained model')
parser.add_argument('--eps', type=float, default=1e-8, help='adamw_epsilon')
parser.add_argument('--optim', type=str, default="adamw", help='Optimizer')
parser.add_argument('--embed_size', type=int, default=8, help='Embedding Size')

# Polar related
parser.add_argument('--beta', type=float, default=0.5,
                    help='Negative sample margin')
parser.add_argument('--vmf_margin', type=float,
                    default=0.5, help='Margin for VMF loss')
parser.add_argument('--c', type=float, default=0.5,
                    help='parameter c for welsch loss')
parser.add_argument('--is_multi_parent', type=bool,
                    default=True, help='If it is a multi parent taxonomy')
parser.add_argument('--geometric_weight', type=float,
                    default=0.5, help='Importance of Geometric Loss')
parser.add_argument('--probabilistic_weight', type=float,
                    default=0.5, help='Importance of Probabilistic Loss')
parser.add_argument(
    '--svgd_weight', help='Importance of Svgd Loss', type=float)
parser.add_argument('--kappa_align', help='Kappa Alignment', type=float)
parser.add_argument('--kappa_repel', help='Kappa repulsion', type=float)
parser.add_argument('--svgd_kernel', help='SVGD kernel')


# Others
parser.add_argument('--cuda', type=bool, default=True,
                    help='use cuda for training')
parser.add_argument('--gpu_id', type=int, default=0, help='which gpu')
parser.add_argument('--seed', type=int, default=20,
                    help="seed for random generators")
parser.add_argument('--method', type=str, default='normal',
                    help='Ablations on method')
parser.add_argument('--path', type=str,
                    default='../your/path/here', help='path to checkpoint')
parser.add_argument('--model', type=str, default='bert', help='PLM Used')
parser.add_argument('--accumulation_steps', type=int, default=1)

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
                    help='If mu needs to be learned during training')
parser.add_argument('--learn_kappa', type=int, default=1,
                    help='If kappa parameter of VMF needs to be learned during training')


start_time = time.time()
print("Start time at : ")
print_local_time()

args = parser.parse_args()
# os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu_id)

args.cuda = True  # inference

torch.cuda.set_device(args.gpu_id)

print(args)
# set_seed(args.seed)

# create_mag_data(args)
exp = Experiments(args)
# exp.visualize_angle_distributions(
#     tag='test', path='/home/nitish/Polartaxo/result/wordnet_verb/train/svgd_kappa_align_repel/experiment_wordnet_verb_14_35_512_0.5_256_0.7_0.4_0.3_0.0_0.0.checkpoint')
# exp.level_wise_prediction(
#     tag='test', path='/home/nitish/Polartaxo/result/mesh/train/experiment_3_varied_geometric_weight/experiment_bert_mesh_2_60_1024_0.5_128_0.5_0.5.')
# exp.predict(tag='test', path='/home/nitish/Polartaxo/result/mesh/train/experiment_3_varied_geometric_weight/experiment_bert_mesh_2_60_1024_0.5_128_0.8_0.5.')
exp.predict_multimodal(
    tag='test', path='/home/nitish/Polartaxo/final_result/birds/independent_seed_runs/experiment_birds_20_0.5_512_0.7_0.4_0.3_4.5_2.5_vmf_theta_0.1_20.pt')
