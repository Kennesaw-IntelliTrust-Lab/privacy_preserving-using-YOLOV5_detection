import torch
from torch.optim import Adam
import os

# YOLOv5 imports
from train import parse_opt, main as yolo_train

# Path to your pruned model
weights_path = '/mnt/bst/hxu10/hxu10/chanti/yolov5/runs/train/exp13/weights/pruned_pytorch_20.pt'

# Fine-tuning hyperparameters
epochs = 20
batch_size = 16
learning_rate = 1e-5

# YOLOv5 train function call
if __name__ == '__main__':
    opt = parse_opt()
    opt.weights = weights_path
    opt.data = '/mnt/bst/hxu10/hxu10/chanti/dataset/data.yaml'
    opt.epochs = epochs
    opt.batch_size = batch_size
    opt.lr0 = learning_rate
    opt.project = 'runs/fine_tune'
    opt.name = 'fine_tuned_model'
    
    # Run YOLOv5 training (fine-tuning)
    yolo_train(opt)
