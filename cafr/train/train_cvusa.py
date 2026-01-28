import os
import time
import math
import shutil
import sys
import torch
import pickle
from dataclasses import dataclass
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup
current_file_path = os.path.abspath(__file__)
project_root_path = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(project_root_path)
from cafr_base.dataset.cvusa_random import CVUSADatasetEval, CVUSADatasetTrain
from cafr_base.transforms import get_transforms_train, get_transforms_val
from cafr_base.utils import setup_system, Logger
from cafr_base.trainer_area import train
from cafr_base.evaluate.cvusa_and_cvact import evaluate, calc_sim
from cafr_base.loss import InfoNCE
from cafr_base.model import RadioModel

@dataclass
class Configuration:

    # Model
    model = 'radio_gem_cafr'
    # backbone
    backbone_arch = 'radio_v2.5-h'
    pretrained = True

    layer1 = 0
    use_cls = True
    norm_descs = True

    # Aggregator 
    agg_arch = 'GeM'  
    agg_config = {}
    apcm_config = {'embed_dim': 1280,
                   'global_dim': 1280,
                   'num_heads': 4,
                   'dropout': 0.4,
                   'max_size': 12,
                   'levels':4
                   }

    # Override model image size
    img_size: int = 384
    new_hight = 384
    new_width = 384

    # Training
    mixed_precision: bool = True
    seed = 1
    epochs: int = 24
    batch_size: int = 24  # keep in mind real_batch_size = 2 * batch_size
    verbose: bool = True
    gpu_ids = [0]
    polar_trans = False

    # Similarity Sampling
    custom_sampling: bool = True  # use custom sampling instead of random
    gps_sample: bool = True  # use gps sampling
    sim_sample: bool = True  # use similarity sampling
    neighbour_select: int = 64  # max selection size from pool 64
    neighbour_range: int = 128  # pool size for selection 128
    gps_dict_path: str = "/root/autodl-tmp/cross_view/cvusa/gps_dict.pkl"  # path to pre-computed distances

    # Eval
    batch_size_eval: int = 24
    eval_every_n_epoch: int = 1  # eval every n Epoch
    normalize_features: bool = True

    # Optimizer
    clip_grad = 100.                   # None | float
    decay_exclue_bias: bool = False
    grad_checkpointing: bool = False   # Gradient Checkpointing
    use_sgd = True

    # Loss
    label_smoothing: float = 0.1

    # Learning Rate
    lr: float = 0.005  # 1 * 10^-4 for ViT | 1 * 10^-3 for CNN   0.0002 for adam, 0.05 for sgd (needs to change according to batch size)
    scheduler: str = "cosine"  # "polynomial" | "cosine" | "constant" | None
    warmup_epochs: int = 1
    lr_end: float = 0.0001  # only for "polynomial"

    # Dataset
    data_folder = "/root/autodl-tmp/cross_view/cvusa"

    # Augment Images
    prob_rotate: float = 0.75  # rotates the sat image and ground images simultaneously
    prob_flip: float = 0.5  # flipping the sat image and ground images simultaneously

    # Savepath for model checkpoints
    model_path: str = "./cvusa"

    # Eval before training
    zero_shot: bool = False

    # Checkpoint to start from
    # checkpoint_start = '/root/autodl-tmp/code/cafr/train/cvusa/gem_poscode/2026-01-19_151257/weights_e24_67.0869.pth'
    checkpoint_start = None
    # set num_workers to 0 if on Windows
    num_workers: int = 12

    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # for better performance
    cudnn_benchmark: bool = True

    # make cudnn deterministic
    cudnn_deterministic: bool = False


#-----------------------------------------------------------------------------#
# Train Config                                                                #
#-----------------------------------------------------------------------------#

config = Configuration()
torch.cuda.set_device(config.gpu_ids[0])

if __name__ == '__main__':
    setup_system(seed=config.seed,
                 cudnn_benchmark=config.cudnn_benchmark,
                 cudnn_deterministic=config.cudnn_deterministic)

    model_path = "{}/{}/{}".format(config.model_path,
                                   config.model,
                                   time.strftime("%Y-%m-%d_%H%M%S"))
    config.outpath = model_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    shutil.copyfile(os.path.abspath(__file__), "{}/train.py".format(model_path))
    moudles_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'cafr_base')
    shutil.copyfile(os.path.join(moudles_path,'trainer_area.py'), "{}/trainer.py".format(model_path))
    shutil.copyfile(os.path.join(os.path.join(moudles_path,'dataset'),'cvusa_random.py'), "{}/dataset.py".format(model_path))
    shutil.copyfile(os.path.join(moudles_path,'model_weight_area.py'), "{}/model.py".format(model_path))
    shutil.copyfile(os.path.join(os.path.join(moudles_path,'evaluate'),'cvusa_and_cvact.py'), "{}/cvusa_and_cvact.py".format(model_path))
    # Redirect print to both console and log file
    sys.stdout = Logger(os.path.join(model_path, 'log.txt'))



    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))

    print("\nModel: {}".format(config.model))

    model = RadioModel(model_name=config.model,
                       pretrained=config.pretrained,
                       img_size=config.img_size, backbone_arch=config.backbone_arch, agg_arch=config.agg_arch,
                       agg_config=config.agg_config,layer=config.layer1,pos_config=config.apcm_config)
    print(model)

    data_config = model.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = config.img_size

    image_size_sat = (img_size, img_size)

    # new_width = config.img_size * 2
    # new_hight = round((224 / 1232) * new_width)
    new_width = config.new_width
    new_hight = config.new_hight
    img_size_ground = (new_hight, new_width)

    # Activate gradient checkpointing
    if config.grad_checkpointing:
        model.set_grad_checkpointing(True)

    # Load pretrained Checkpoint
    if config.checkpoint_start is not None:
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)
        model.load_state_dict(model_state_dict, strict=False)

    # Data parallel
    print("GPUs available:", torch.cuda.device_count())
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
        model = model.to(config.device)
        print("Model: DataParallel")
    else:
        # # Model to device
        config.device = f'cuda:{config.gpu_ids[0]}'
        model = model.to(config.device)
        print(f"Model: cuda:{config.device}")
    # # Model to device
    # model = model.to(config.device)

    print("\nImage Size Sat:", image_size_sat)
    print("Image Size Ground:", img_size_ground)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std))


    #-----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    #-----------------------------------------------------------------------------#

    # Transforms
    sat_transforms_train, ground_transforms_train = get_transforms_train(image_size_sat,
                                                                   img_size_ground,
                                                                   mean=mean,
                                                                   std=std,
                                                                   )
    # Eval
    sat_transforms_val, ground_transforms_val = get_transforms_val(image_size_sat,
                                                                   img_size_ground,
                                                                   mean=mean,
                                                                   std=std,
                                                                   )

    # Train
    train_dataset = CVUSADatasetTrain(data_folder=config.data_folder ,
                                      transforms_query=ground_transforms_train,
                                      transforms_reference=sat_transforms_train,
                                      prob_flip=config.prob_flip,
                                      prob_rotate=config.prob_rotate,
                                      shuffle_batch_size=config.batch_size
                                      )

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  shuffle=not config.custom_sampling,
                                  pin_memory=True)

    # Eval

    # Reference Satellite Images
    reference_dataset_test = CVUSADatasetEval(data_folder=config.data_folder ,
                                              split="test",
                                              img_type="reference",
                                              transforms=sat_transforms_val,
                                              )

    reference_dataloader_test = DataLoader(reference_dataset_test,
                                           batch_size=config.batch_size_eval,
                                           num_workers=config.num_workers,
                                           shuffle=False,
                                           pin_memory=True)


    # Query Ground Images Test
    query_dataset_test = CVUSADatasetEval(data_folder=config.data_folder ,
                                          split="test",
                                          img_type="query",
                                          transforms=ground_transforms_val,
                                          )

    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)


    print("Reference Images Test:", len(reference_dataset_test))
    print("Query Images Test:", len(query_dataset_test))


    #-----------------------------------------------------------------------------#
    # GPS Sample                                                                  #
    #-----------------------------------------------------------------------------#
    if config.gps_sample:
        with open(config.gps_dict_path, "rb") as f:
            sim_dict = pickle.load(f)
    else:
        sim_dict = None

    #-----------------------------------------------------------------------------#
    # Sim Sample                                                                  #
    #-----------------------------------------------------------------------------#

    if config.sim_sample:

        # Query Ground Images Train for simsampling
        query_dataset_train = CVUSADatasetEval(data_folder=config.data_folder ,
                                               split="train",
                                               img_type="query",
                                               transforms=ground_transforms_val,
                                               )

        query_dataloader_train = DataLoader(query_dataset_train,
                                            batch_size=config.batch_size_eval,
                                            num_workers=config.num_workers,
                                            shuffle=False,
                                            pin_memory=True)


        reference_dataset_train = CVUSADatasetEval(data_folder=config.data_folder ,
                                                   split="train",
                                                   img_type="reference",
                                                   transforms=sat_transforms_val,
                                                   )

        reference_dataloader_train = DataLoader(reference_dataset_train,
                                                batch_size=config.batch_size_eval,
                                                num_workers=config.num_workers,
                                                shuffle=False,
                                                pin_memory=True)


        print("\nReference Images Train:", len(reference_dataset_train))
        print("Query Images Train:", len(query_dataset_train))


    #-----------------------------------------------------------------------------#
    # Loss                                                                        #
    #-----------------------------------------------------------------------------#

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    loss_function = InfoNCE(loss_function=loss_fn,
                            device=config.device,
                            )

    if config.mixed_precision:
        scaler = GradScaler(init_scale=2.**10)
    else:
        scaler = None

    #-----------------------------------------------------------------------------#
    # optimizer                                                                   #
    #-----------------------------------------------------------------------------#
    if config.decay_exclue_bias:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        # optimizer = torch.optim.AdamW(optimizer_parameters, lr=config.lr,weight_decay=0.005)
        optimizer = torch.optim.AdamW(optimizer_parameters, lr=config.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    if config.use_sgd:
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)

    #-----------------------------------------------------------------------------#
    # Scheduler                                    #
    #-----------------------------------------------------------------------------#

    train_steps = len(train_dataloader) * config.epochs
    warmup_steps = len(train_dataloader) * config.warmup_epochs

    if config.scheduler == "polynomial":
        print("\nScheduler: polynomial - max LR: {} - end LR: {}".format(config.lr, config.lr_end))
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                              num_training_steps=train_steps,
                                                              lr_end = config.lr_end,
                                                              power=1.5,
                                                              num_warmup_steps=warmup_steps)

    elif config.scheduler == "cosine":
        print("\nScheduler: cosine - max LR: {}".format(config.lr))
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_training_steps=train_steps,
                                                    num_warmup_steps=warmup_steps)

    elif config.scheduler == "constant":
        print("\nScheduler: constant - max LR: {}".format(config.lr))
        scheduler = get_constant_schedule_with_warmup(optimizer,
                                                       num_warmup_steps=warmup_steps)

    else:
        scheduler = None

    print("Warmup Epochs: {} - Warmup Steps: {}".format(str(config.warmup_epochs).ljust(2), warmup_steps))
    print("Train Epochs:  {} - Train Steps:  {}".format(config.epochs, train_steps))

    #-----------------------------------------------------------------------------#
    # Zero Shot                                                                   #
    #-----------------------------------------------------------------------------#
    if config.zero_shot:
        print("\n{}[{}]{}".format(30*"-", "Zero Shot", 30*"-"))


        r1_test = evaluate(config=config,
                           model=model,
                           reference_dataloader=reference_dataloader_test,
                           query_dataloader=query_dataloader_test,
                           ranks=[1, 5, 10],
                           step_size=1000,
                           cleanup=True)

        if config.sim_sample:
            r1_train, sim_dict = calc_sim(config=config,
                                          model=model,
                                          reference_dataloader=reference_dataloader_train,
                                          query_dataloader=query_dataloader_train,
                                          ranks=[1, 5, 10],
                                          step_size=1000,
                                          cleanup=True)
    #-----------------------------------------------------------------------------#
    # Shuffle                                                                     #
    #-----------------------------------------------------------------------------#
    if config.custom_sampling:
        train_dataloader.dataset.shuffle(sim_dict,
                                         neighbour_select=config.neighbour_select,
                                         neighbour_range=config.neighbour_range)
    #-----------------------------------------------------------------------------#
    # Train                                                                       #
    #-----------------------------------------------------------------------------#
    start_epoch = 0
    best_score = 0

    for epoch in range(1, config.epochs+1):

        print("\n{}[{}/Epoch: {}]{}".format(30*"-",time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),  epoch, 30*"-"))

        train_loss,pos_loss = train(config,
                           model,epoch,
                           dataloader=train_dataloader,
                           loss_function=loss_function,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           scaler=scaler)
        # train_loss = 1

        print("Epoch: {}, Train Loss = {:.3f},pos Loss = {:.3f}, Lr = {:.6f}".format(epoch,
                                                                   train_loss,pos_loss,
                                                                   optimizer.param_groups[0]['lr']))
        # evaluate
        if (epoch % config.eval_every_n_epoch == 0 and epoch != 0) or epoch == config.epochs:

            print("\n{}[{}]{}".format(30*"-", "Evaluate", 30*"-"))

            r1_test = evaluate(config=config,
                               model=model,
                               reference_dataloader=reference_dataloader_test,
                               query_dataloader=query_dataloader_test,
                               ranks=[1, 5, 10],
                               step_size=1000,
                               cleanup=True)
            if config.sim_sample:
                r1_train,sim_dict = calc_sim(config=config,
                                              model=model,
                                              reference_dataloader=reference_dataloader_train,
                                              query_dataloader=query_dataloader_train,
                                              ranks=[1, 5, 10],
                                              step_size=1000,
                                              cleanup=True)

            if r1_test > best_score:

                best_score = r1_test
                if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
                    state_dict_cpu = {k: v.to('cpu') for k, v in model.module.state_dict().items()}
                    torch.save(state_dict_cpu, '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))
                else:
                    state_dict_cpu = {k: v.to('cpu') for k, v in model.state_dict().items()}
                    torch.save(state_dict_cpu, '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))

            if config.custom_sampling:
                train_dataloader.dataset.shuffle(sim_dict,
                                                 neighbour_select=config.neighbour_select,
                                                 neighbour_range=config.neighbour_range)

            if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
                state_dict_cpu = {k: v.to('cpu') for k, v in model.module.state_dict().items()}
                torch.save(state_dict_cpu, '{}/weights_end.pth'.format(model_path))
            else:
                state_dict_cpu = {k: v.to('cpu') for k, v in model.state_dict().items()}
                torch.save(state_dict_cpu, '{}/weights_end.pth'.format(model_path))    
