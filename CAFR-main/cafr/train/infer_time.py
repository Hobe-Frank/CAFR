import os
import time
import math
import shutil
import sys
import torch
import pickle
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from dataclasses import dataclass
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, \
    get_cosine_schedule_with_warmup
from torch.amp import autocast

current_file_path = os.path.abspath(__file__)
project_root_path = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(project_root_path)
from cafr_base.dataset.cvnaf import CVNAFDatasetEval, CVNAFDatasetTrain, CVNAFDatasetSim
from cafr_base.cafr_transforms import get_transforms_train, get_transforms_val
from cafr_base.utils import setup_system, Logger
from cafr_base.trainer_area import train, predict
from cafr_base.evaluate.cvnaf import evaluate, calc_sim, calc_sim_no_r1
from cafr_base.loss import InfoNCE
from cafr_base.model import RadioModel


def benchmark_single_query_inference(config, model, query_img, reference_features, reference_labels, 
                                      warmup_iterations=10, test_iterations=100):
    """
    Benchmark single query inference time and GPU memory usage after warmup.
    """
    model.eval()
    
    if query_img.dim() == 3:
        query_img = query_img.unsqueeze(0)
    
    query_img = query_img.to(config.device)
    reference_features = reference_features.to(config.device)
    
    print("\n=== Single Query Inference Benchmark ===")
    print(f"Warmup iterations: {warmup_iterations}")
    print(f"Test iterations: {test_iterations}")
    print(f"Reference database size: {len(reference_features)}")
    
    # Reset memory stats before warmup
    torch.cuda.reset_peak_memory_stats(config.device)
    torch.cuda.empty_cache()
    
    # Record baseline memory
    baseline_allocated = torch.cuda.memory_allocated(config.device) / 1024**3
    baseline_reserved = torch.cuda.memory_reserved(config.device) / 1024**3
    
    print(f"\nBaseline GPU Memory:")
    print(f"  Allocated: {baseline_allocated:.3f} GB")
    print(f"  Reserved: {baseline_reserved:.3f} GB")
    
    # Warmup phase
    print("\nWarming up...")
    with torch.no_grad(), autocast('cuda'):
        for _ in range(warmup_iterations):
            _, query_feat = model(query_img)
            if config.normalize_features:
                query_feat = F.normalize(query_feat, dim=-1)
            sim = query_feat @ reference_features.T
            _ = torch.argmax(sim, dim=1)
    
    torch.cuda.synchronize()
    
    # Reset peak memory stats after warmup
    torch.cuda.reset_peak_memory_stats(config.device)
    
    # Timing and memory tracking phase
    print("Running benchmark...")
    feature_extraction_times = []
    retrieval_times = []
    total_times = []
    memory_snapshots = []
    
    with torch.no_grad(), autocast('cuda'):
        for i in tqdm(range(test_iterations), desc="Benchmarking"):
            # Reset peak for this iteration
            torch.cuda.reset_peak_memory_stats(config.device)
            
            # Time feature extraction
            torch.cuda.synchronize()
            start_feature = time.perf_counter()
            
            _, query_feat = model(query_img)
            if config.normalize_features:
                query_feat = F.normalize(query_feat, dim=-1)
            
            torch.cuda.synchronize()
            end_feature = time.perf_counter()
            feature_time = (end_feature - start_feature) * 1000
            
            # Memory after feature extraction
            feat_allocated = torch.cuda.memory_allocated(config.device) / 1024**3
            feat_peak = torch.cuda.max_memory_allocated(config.device) / 1024**3
            
            # Time similarity computation and retrieval
            torch.cuda.synchronize()
            start_retrieval = time.perf_counter()
            
            sim = query_feat @ reference_features.T
            top_idx = torch.argmax(sim, dim=1)
            
            torch.cuda.synchronize()
            end_retrieval = time.perf_counter()
            retrieval_time = (end_retrieval - start_retrieval) * 1000
            
            # Memory after retrieval
            total_allocated = torch.cuda.memory_allocated(config.device) / 1024**3
            total_peak = torch.cuda.max_memory_allocated(config.device) / 1024**3
            
            total_time = feature_time + retrieval_time
            
            feature_extraction_times.append(feature_time)
            retrieval_times.append(retrieval_time)
            total_times.append(total_time)
            memory_snapshots.append({
                'allocated': total_allocated,
                'peak': total_peak
            })
    
    # Calculate statistics
    feature_extraction_times = np.array(feature_extraction_times)
    retrieval_times = np.array(retrieval_times)
    total_times = np.array(total_times)
    
    # Memory statistics
    allocated_mem = np.array([m['allocated'] for m in memory_snapshots])
    peak_mem = np.array([m['peak'] for m in memory_snapshots])
    
    results = {
        'feature_extraction': {
            'mean': np.mean(feature_extraction_times),
            'std': np.std(feature_extraction_times),
            'min': np.min(feature_extraction_times),
            'max': np.max(feature_extraction_times),
            'median': np.median(feature_extraction_times)
        },
        'retrieval': {
            'mean': np.mean(retrieval_times),
            'std': np.std(retrieval_times),
            'min': np.min(retrieval_times),
            'max': np.max(retrieval_times),
            'median': np.median(retrieval_times)
        },
        'total': {
            'mean': np.mean(total_times),
            'std': np.std(total_times),
            'min': np.min(total_times),
            'max': np.max(total_times),
            'median': np.median(total_times)
        },
        'memory': {
            'baseline_allocated_gb': baseline_allocated,
            'baseline_reserved_gb': baseline_reserved,
            'allocated_mean_gb': np.mean(allocated_mem),
            'allocated_max_gb': np.max(allocated_mem),
            'peak_mean_gb': np.mean(peak_mem),
            'peak_max_gb': np.max(peak_mem),
            'inference_overhead_gb': np.mean(peak_mem) - baseline_allocated
        }
    }
    
    # Print results
    print("\n=== Benchmark Results (after warmup) ===")
    print(f"\nFeature Extraction:")
    print(f"  Mean: {results['feature_extraction']['mean']:.3f} ms")
    print(f"  Std:  {results['feature_extraction']['std']:.3f} ms")
    print(f"  Min:  {results['feature_extraction']['min']:.3f} ms")
    print(f"  Max:  {results['feature_extraction']['max']:.3f} ms")
    print(f"  Median: {results['feature_extraction']['median']:.3f} ms")
    
    print(f"\nRetrieval (Similarity + Argmax):")
    print(f"  Mean: {results['retrieval']['mean']:.3f} ms")
    print(f"  Std:  {results['retrieval']['std']:.3f} ms")
    print(f"  Min:  {results['retrieval']['min']:.3f} ms")
    print(f"  Max:  {results['retrieval']['max']:.3f} ms")
    print(f"  Median: {results['retrieval']['median']:.3f} ms")
    
    print(f"\nTotal Inference Time:")
    print(f"  Mean: {results['total']['mean']:.3f} ms")
    print(f"  Std:  {results['total']['std']:.3f} ms")
    print(f"  Min:  {results['total']['min']:.3f} ms")
    print(f"  Max:  {results['total']['max']:.3f} ms")
    print(f"  Median: {results['total']['median']:.3f} ms")
    print(f"  FPS (based on mean): {1000.0 / results['total']['mean']:.2f}")
    
    print(f"\nGPU Memory Usage:")
    print(f"  Baseline Allocated: {results['memory']['baseline_allocated_gb']:.3f} GB")
    print(f"  Baseline Reserved: {results['memory']['baseline_reserved_gb']:.3f} GB")
    print(f"  Inference Allocated (mean): {results['memory']['allocated_mean_gb']:.3f} GB")
    print(f"  Inference Allocated (max): {results['memory']['allocated_max_gb']:.3f} GB")
    print(f"  Inference Peak (mean): {results['memory']['peak_mean_gb']:.3f} GB")
    print(f"  Inference Peak (max): {results['memory']['peak_max_gb']:.3f} GB")
    print(f"  Inference Overhead: {results['memory']['inference_overhead_gb']:.3f} GB")
    
    return results


def run_inference_benchmark(config, model, query_dataloader, reference_dataloader, 
                            warmup_iterations=10, test_iterations=10, 
                            use_cached=False, cache_dir='./benchmark_cache'):
    """
    Wrapper function to run the benchmark with data loaders.
    Supports caching reference features for reuse.
    
    Args:
        config: Configuration object
        model: The model to benchmark
        query_dataloader: DataLoader for query images
        reference_dataloader: DataLoader for reference images
        warmup_iterations: Number of warmup iterations
        test_iterations: Number of test iterations
        use_cached: Whether to use cached reference features if available
        cache_dir: Directory to save/load cached features
    """
    os.makedirs(cache_dir, exist_ok=True)
    ref_feat_path = os.path.join(cache_dir, 'reference_features.pth')
    ref_label_path = os.path.join(cache_dir, 'reference_labels.pth')
    
    # Load or compute reference features
    if use_cached and os.path.exists(ref_feat_path) and os.path.exists(ref_label_path):
        print("\n=== Loading Cached Reference Database ===")
        reference_features = torch.load(ref_feat_path).to(config.device)
        reference_labels = torch.load(ref_label_path).to(config.device)
        print(f"Loaded reference features: {reference_features.shape}")
    else:
        print("\n=== Computing Reference Database ===")
        reference_features, reference_labels = predict(config, model, reference_dataloader)
        
        # Save for future use
        print("\n=== Saving Reference Database ===")
        torch.save(reference_features.cpu(), ref_feat_path)
        torch.save(reference_labels.cpu(), ref_label_path)
        print(f"Saved to {cache_dir}")
    
    # Get a single query image
    print("\n=== Loading Query Image ===")
    query_img, query_label = next(iter(query_dataloader))
    query_img = query_img[0]  # Take first image from batch
    
    print(f"Query image shape: {query_img.shape}")
    print(f"Query label: {query_label[0].item()}")
    
    # Run benchmark
    results = benchmark_single_query_inference(
        config=config,
        model=model,
        query_img=query_img,
        reference_features=reference_features,
        reference_labels=reference_labels,
        warmup_iterations=warmup_iterations,
        test_iterations=test_iterations
    )
    
    return results


@dataclass
class Configuration:
    # Model
    model = 'radio_gem_cafr'
    # backbone
    backbone_arch = 'radio_v2.5-h'
    pretrained = True
    layer1 = -1
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
                   'levels': 4
                   }

    # Override model image size
    crop_p = 1
    img_size: int = 384

    # Training
    mixed_precision: bool = True
    seed = 1
    epochs: int = 24
    batch_size: int = 24
    verbose: bool = True
    gpu_ids = [0]
    
    TRAIN_CITIES = [
        'Tuscany',  
        'Salzburg',  
        'Miami',  
        'NewYork'  
    ]

    TEST_CITIES = [
        'eval_all'
    ]
    
    # Similarity Sampling
    custom_sampling: bool = True
    gps_sample: bool = True
    sim_sample: bool = True
    neighbour_select: int = 64
    neighbour_range: int = 128
    gps_dict_path: str = f"/root/autodl-tmp/cross_view/CVNAF/gps_dict_{len(TRAIN_CITIES)}_cities.pkl"

    # Eval
    batch_size_eval: int = 48
    eval_every_n_epoch: int = 1
    normalize_features: bool = True

    # Optimizer
    clip_grad = 100.
    decay_exclue_bias: bool = False
    grad_checkpointing: bool = False
    use_sgd = True

    # Loss
    label_smoothing: float = 0.1

    # Learning Rate
    lr: float = 0.001
    scheduler: str = "cosine"
    warmup_epochs: int = 1
    lr_end: float = 0.0001

    # Dataset
    data_folder = '/root/autodl-tmp/cross_view/CVNAF'

    # Augment Images
    prob_rotate: float = 0.75
    prob_flip: float = 0.5

    # Savepath for model checkpoints
    model_path: str = "./cvnaf"

    # Eval before training
    zero_shot: bool = False

    # Checkpoint to start from
    checkpoint_start = None

    # set num_workers to 0 if on Windows
    num_workers: int = 12

    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # for better performance
    cudnn_benchmark: bool = True

    # make cudnn deterministic
    cudnn_deterministic: bool = False


config = Configuration()

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
    moudles_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cafr_base')
    shutil.copyfile(os.path.join(moudles_path, 'trainer_area.py'), "{}/trainer.py".format(model_path))
    shutil.copyfile(os.path.join(os.path.join(moudles_path, 'dataset'), 'cvnaf_perturb.py'),
                    "{}/dataset.py".format(model_path))
    shutil.copyfile(os.path.join(moudles_path, 'model.py'), "{}/model.py".format(model_path))
    shutil.copyfile(os.path.join(os.path.join(moudles_path, 'evaluate'), 'cvnaf.py'),
                    "{}/evaluate.py".format(model_path))
    
    # Redirect print to both console and log file
    sys.stdout = Logger(os.path.join(model_path, 'log.txt'))

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
    print("\nModel: {}".format(config.model))
    
    layers_to_freeze = 20
    model = RadioModel(model_name=config.model,
                       pretrained=config.pretrained,
                       img_size=config.img_size, 
                       backbone_arch=config.backbone_arch, 
                       agg_arch=config.agg_arch,
                       agg_config=config.agg_config, 
                       layer=config.layer1, 
                       pos_config=config.apcm_config)
    
    print(model)
    print('layers_to_freeze=', layers_to_freeze)
    
    model_preprocess = model.model.preprocessor
    data_config = model.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = config.img_size

    image_size_sat = (img_size, img_size)
    new_width = config.img_size
    new_hight = config.img_size
    img_size_ground = (new_hight, new_width)

    if config.grad_checkpointing:
        model.set_grad_checkpointing(True)

    if config.checkpoint_start is not None:
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)
        model.load_state_dict(model_state_dict, strict=False)

    print("GPUs available:", torch.cuda.device_count())
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
        model = model.to(config.device)
        print("Model: DataParallel")
    else:
        config.device = f'cuda:{config.gpu_ids[0]}'
        model = model.to(config.device)
        print(f"Model: {config.device}")

    print("\nImage Size Sat:", image_size_sat)
    print("Image Size Ground:", img_size_ground)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std))

    sat_transforms_val, ground_transforms_val = get_transforms_val(image_size_sat,
                                                                   img_size_ground,
                                                                   model_preprocess,
                                                                   mean=mean,
                                                                   std=std,
                                                                   )

    for test_city in config.TEST_CITIES:
        print(f'Evaluating test_city: ------------{test_city}------------')

        # Reference Satellite Images
        reference_dataset_test = CVNAFDatasetEval(data_folder=config.data_folder,
                                                  split="test",
                                                  img_type="reference",
                                                  transforms=sat_transforms_val,
                                                  train_cities=[test_city],
                                                  test_cities=[test_city]
                                                  )

        reference_dataloader_test = DataLoader(reference_dataset_test,
                                               batch_size=config.batch_size_eval,
                                               num_workers=config.num_workers,
                                               shuffle=False,
                                               pin_memory=True)

        # Query Ground Images Test
        query_dataset_test = CVNAFDatasetEval(data_folder=config.data_folder,
                                              split="test",
                                              img_type="query",
                                              transforms=ground_transforms_val,
                                              train_cities=[test_city],
                                              test_cities=[test_city]
                                              )

        query_dataloader_test = DataLoader(query_dataset_test,
                                           batch_size=config.batch_size_eval,
                                           num_workers=config.num_workers,
                                           shuffle=False,
                                           pin_memory=True)

        # Run benchmark with caching support
        # First run: use_cached=False to compute and save
        # Subsequent runs: use_cached=True to load cached features
        results = run_inference_benchmark(
            config=config,
            model=model,
            query_dataloader=query_dataloader_test,
            reference_dataloader=reference_dataloader_test,
            warmup_iterations=50,
            test_iterations=100,
            use_cached=True,  # Set to True to reuse cached features
            cache_dir=f'benchmark_cache_{test_city}'
        )