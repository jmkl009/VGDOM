import numpy as np
import os
import torch
import torch.nn as nn

from constants import Constants
from datasets import load_vistext_data
from evaluate import evaluate
from models import VisTextGCN, BboxOnlyModel
from train import train_vistext_model, evaluate_vistext_model
from datasets import load_data, GraphConstuctor

from utils import cmdline_args_parser, print_and_log, set_all_seeds

# tree object from stanfordnlp/treelstm
class Tree(object):
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()
        
    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self, '_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self, '_depth'):
            return self._depth
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth



parser = cmdline_args_parser()
args = parser.parse_args()

device = torch.device('cuda:%d' % args.device if torch.cuda.is_available() else 'cpu')
# set_all_seeds(Constants.SEED)

N_CLASSES = Constants.N_CLASSES
CLASS_NAMES = Constants.CLASS_NAMES
IMG_HEIGHT = Constants.IMG_HEIGHT
DATA_DIR = Constants.DATA_DIR
SPLIT_DIR = Constants.SPLIT_DIR
OUTPUT_DIR = Constants.OUTPUT_DIR
# NOTE: if same hyperparameter configuration is run again, previous log file and saved model will be overwritten

EVAL_INTERVAL = 1 # Number of Epochs after which model is evaluated while training
NUM_WORKERS = args.num_workers # multithreaded data loading

CV_FOLD = args.cv_fold
FOLD_DIR = '%s/Fold-%d' % (SPLIT_DIR, CV_FOLD)
if CV_FOLD == -1:
    FOLD_DIR = SPLIT_DIR # use files from SPLIT_DIR

train_img_ids = np.loadtxt('%s/train_imgs.txt' % FOLD_DIR, str)
val_img_ids = np.loadtxt('%s/val_imgs.txt' % FOLD_DIR, str)
test_img_ids = np.loadtxt('%s/test_imgs.txt' % FOLD_DIR, str)

# for calculating domainwise and macro accuracy if below files are available (optional)
webpage_info_file = '%s/webpage_info.csv' % FOLD_DIR
webpage_info = None
if os.path.isfile(webpage_info_file):
    webpage_info = np.loadtxt(webpage_info_file, str, delimiter=',', skiprows=1) # (img_id, domain) values

test_domains_file = '%s/test_domains.txt' % FOLD_DIR
test_domains = None
if os.path.isfile(test_domains_file):
    test_domains = np.loadtxt(test_domains_file, str)

########## HYPERPARAMETERS ##########
N_EPOCHS = args.n_epochs
BACKBONE = 'CNN-BiLSTM'
TRAINABLE_CONVNET = not args.freeze_convnet
LEARNING_RATE = args.learning_rate
BATCH_SIZE = args.batch_size
HIDDEN_DIM = args.hidden_dim
ROI_OUTPUT = (args.roi, args.roi)
USE_BBOX_FEAT = args.bbox_feat
BBOX_HIDDEN_DIM = args.bbox_hidden_dim if USE_BBOX_FEAT else 0
USE_ADDITIONAL_FEAT = args.additional_feat
WEIGHT_DECAY = args.weight_decay
DROP_PROB = args.drop_prob

params = '%s lr-%.0e batch-%d hd-%d roi-%d bbf-%d bbhd-%d af-%d wd-%.0e dp-%.1f' % (BACKBONE, LEARNING_RATE,
    BATCH_SIZE, HIDDEN_DIM, ROI_OUTPUT[0], USE_BBOX_FEAT, BBOX_HIDDEN_DIM,
    USE_ADDITIONAL_FEAT, WEIGHT_DECAY, DROP_PROB)
results_dir = '%s/%s' % (OUTPUT_DIR, params)
fold_wise_acc_file = '%s/fold_wise_acc.csv' % results_dir

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

print('\n%s Training on Fold-%s %s' % ('*'*20, CV_FOLD, '*'*20))
########## DATA LOADERS ##########
train_loader, val_loader, test_loader = load_vistext_data(DATA_DIR, train_img_ids, val_img_ids, test_img_ids, 
                                                  BATCH_SIZE, NUM_WORKERS)
# n_additional_features = train_loader.dataset.n_additional_features
pretrained_word_vectors = train_loader.dataset.pretrained_word_vectors

log_file = '%s/Fold-%s logs.txt' % (results_dir, CV_FOLD)
test_acc_imgwise_file = '%s/Fold-%s test_acc_imgwise.csv' % (results_dir, CV_FOLD)
test_acc_domainwise_file = '%s/Fold-%s test_acc_domainwise.csv' % (results_dir, CV_FOLD)
model_save_file = '%s/Fold-%s saved_model.pth' % (results_dir, CV_FOLD)

print('logs will be saved in \"%s\"' % (log_file))
print_and_log('Backbone Convnet: %s' % (BACKBONE), log_file, 'w')
print_and_log('Trainable Convnet: %s' % (TRAINABLE_CONVNET), log_file)
print_and_log('Learning Rate: %.0e' % (LEARNING_RATE), log_file)
print_and_log('Batch Size: %d' % (BATCH_SIZE), log_file)
print_and_log('Hidden Dim: %d' % (HIDDEN_DIM), log_file)
print_and_log('RoI Pool Output Size: (%d, %d)' % ROI_OUTPUT, log_file)
print_and_log('Use BBox Features: %s' % (USE_BBOX_FEAT), log_file)
print_and_log('BBox Hidden Dim: %d' % (BBOX_HIDDEN_DIM), log_file)
print_and_log('Use Additional Features: %s' % (USE_ADDITIONAL_FEAT), log_file)
print_and_log('Weight Decay: %.0e' % (WEIGHT_DECAY), log_file)
print_and_log('Dropout Probability: %.2f' % (DROP_PROB), log_file)

########## TRAIN MODEL ##########

num_candidates=args.num_candidates
num_rels = GraphConstuctor(num_candidates).num_rels

char_vocab_size = train_loader.dataset.char_vocab_size
tag_vocab_size = train_loader.dataset.tag_vocab_size
CHAR_EMBED_DIM = 8
TAG_EMBED_DIM = 30

model = VisTextGCN(char_vocab_size, tag_vocab_size, N_CLASSES, num_candidates, 
                        num_rels, device, pretrained_word_vectors, HIDDEN_DIM, USE_BBOX_FEAT,
               CHAR_EMBED_DIM, TAG_EMBED_DIM, BBOX_HIDDEN_DIM, TRAINABLE_CONVNET, DROP_PROB, CLASS_NAMES, splitted=False).to(device)
# model.split()
# model = model.half()
# model.gcn = model.gcn.float()
# model = BboxOnlyModel().to(device)
# model.to_parallel()
# model = nn.DataParallel(model)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=1) # No LR Scheduling
criterion = nn.CrossEntropyLoss(reduction='sum').to(device)
# criterion = nn.BCEWithLogitsLoss(reduction='sum').to(device)
if args.retrain:
    print('Loading Saved Model...')
    model.load_state_dict(torch.load(model_save_file))

# def try_train():
#     try:
#         val_acc, gcn_model_save_file = train_vistext_model(model, train_loader, optimizer, scheduler, criterion, N_EPOCHS, device, val_loader, EVAL_INTERVAL,
#                         log_file, model_save_file)
#     except Exception as e: 
#         print(e)
#         return False
#     return val_acc, gcn_model_save_file

# train_result = try_train()
# while not train_result:
#     torch.cuda.empty_cache()
#     train_result = try_train()
# val_acc, gcn_model_save_file = train_result
val_acc, gcn_model_save_file = train_vistext_model(model, train_loader, optimizer, scheduler, criterion, N_EPOCHS, device, val_loader, EVAL_INTERVAL,
                        log_file, model_save_file)

# save_file_delimited = model_save_file.split('/')
# prev_path = "/".join(save_file_delimited[:-1])
# save_file_delimited = save_file_delimited[-1].split('.')
# gcn_model_save_file = (
#        prev_path + "/" + save_file_delimited[0] + '-gcn.' + save_file_delimited[1]
#        if len(save_file_delimited) > 1 else 
#        prev_path + "/" + save_file_delimited[0] + '-gcn' 
#        )

import sys

print('Model Trained! Restoring model to best Eval performance checkpoint...')
model.load_state_dict(torch.load(model_save_file))

_ = evaluate_vistext_model(model, test_loader, device, 1, 'TEST', log_file)

print('Evaluating on test data with best GCN model...')
model.load_state_dict(torch.load(gcn_model_save_file))
_ = evaluate_vistext_model(model, test_loader, device, 1, 'TEST', log_file)
