
# Check for CUDA availability
import argparse
import numpy as np
import torch
import hashlib
import os
from torch.utils.data import DataLoader
import data_processing
from litbox_dataset import LitboxDataset
import json
from welford_torch import Welford
import pprint
from data_processing import save_image
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
g_batch_size = 4

def parse_args():
    parser = argparse.ArgumentParser(description='Litbox Denoiser Training Script')
    parser.add_argument('--compute-stats', action='store_true', help='Compute global dataset stats.')
    parser.add_argument('--cache-location', help='Path to preprocessed feature cache', default='./training_cache')
    data_processing.register_dataset_args(parser, True)
    
    args = parser.parse_args()
    data_processing.validate_dataset_args(args, parser)
    return args

def get_cache_path(args):
    def none_as_empty(s):
        if s is None:
            return ""
        else:
            return s

    # Figure out the cache folder
    ref_hash = hashlib.md5((
        args.reference_location +
        none_as_empty(args.input_a_easy_location) +
        none_as_empty(args.input_b_easy_location) +
        none_as_empty(args.input_a_medium_location) +
        none_as_empty(args.input_b_medium_location) +
        none_as_empty(args.input_a_location) +
        none_as_empty(args.input_b_location) +
        none_as_empty(args.input_albedo_location) +
        none_as_empty(args.input_transmissibility_location)).encode()).hexdigest()
    cache_dir = os.path.join(args.cache_location, ref_hash)
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def create_litbox_raw_data_loader(args):
    data_files = data_processing.get_dataset_files(args.reference_location,
                                                   input_a_easy=args.input_a_easy_location, input_b_easy=args.input_b_easy_location,
                                                   input_a_medium=args.input_a_medium_location, input_b_medium=args.input_b_medium_location,
                                                   input_a_final=args.input_a_location, input_b_final=args.input_b_location,
                                                   albedo=args.input_albedo_location, transmissibility=args.input_transmissibility_location)

    dataset = LitboxDataset({k: v for k, v in {
        'reference': (data_files.get('reference'), 3),
        'albedo': (data_files.get('albedo'), 3),
        'transmissibility': (data_files['transmissibility'], 1),
        'input_a_final': (data_files.get('input_a_final'), 3),
        'input_b_final': (data_files.get('input_b_final'), 3),
        'input_a_easy' : (data_files.get('input_a_easy'), 3),
        'input_b_easy' : (data_files.get('input_b_easy'), 3),
        'input_a_medium' : (data_files.get('input_a_medium'), 3),
        'input_b_medium' : (data_files.get('input_b_medium'), 3)
    }.items() if v[0] is not None})
    return dataset, DataLoader(dataset, batch_size=g_batch_size, shuffle=False, num_workers=4)

def create_litbox_cached_datasets(args, device):
    has_easy = args.input_a_easy_location is not None
    has_medium = args.input_a_medium_location is not None
    cache_dir = get_cache_path(args)

    radiance_easy_location = None
    variance_easy_location = None
    if has_easy:
        radiance_easy_location = cache_dir + '/*_Radiance_Easy.exr'
        variance_easy_location = cache_dir + '/*_Variance_Easy.exr'

    radiance_medium_location = None
    variance_medium_location = None
    if has_medium:
        radiance_medium_location = cache_dir + '/*_Radiance_Medium.exr'
        variance_medium_location = cache_dir + '/*_Variance_Medium.exr'

    radiance_final_location = cache_dir + '/*_Radiance_Final.exr'
    variance_final_location = cache_dir + '/*_Variance_Final.exr'
    density_location = cache_dir + '/*_Density.exr'
    reference_location = cache_dir + '/*_Reference.exr'
    albedo_location = cache_dir + '/*_Albedo.png'

    data_files = data_processing.get_dataset_files(reference_location,
                                                   radiance_easy=radiance_easy_location, variance_easy=variance_easy_location,
                                                   radiance_medium=radiance_medium_location, variance_medium=variance_medium_location,
                                                   radiance_final=radiance_final_location, variance_final=variance_final_location,
                                                   albedo=albedo_location, density=density_location)
    
    split_idx = int(len(data_files['reference']) * (1 - args.test_ratio))

    training_datasets = []
    if has_easy:
        set = LitboxDataset({k: v for k, v in {
            'reference': (data_files['reference'][:split_idx], 3),
            'albedo': (data_files['albedo'][:split_idx], 3),
            'density': (data_files['density'][:split_idx], 1),
            'radiance' : (data_files['radiance_easy'][:split_idx], 3),
            'variance' : (data_files['variance_easy'][:split_idx], 1),
        }.items() if v[0] is not None}, device=device)
        set.name = "Easy"
        training_datasets.append(set)

    if has_medium:
        set =LitboxDataset({k: v for k, v in {
            'reference': (data_files['reference'][:split_idx], 3),
            'albedo': (data_files['albedo'][:split_idx], 3),
            'density': (data_files['density'][:split_idx], 1),
            'radiance' : (data_files['radiance_medium'][:split_idx], 3),
            'variance' : (data_files['variance_medium'][:split_idx], 1),
        }.items() if v[0] is not None}, device=device)
        set.name = "Medium"
        training_datasets.append(set)
    
    set = LitboxDataset({k: v for k, v in {
        'reference': (data_files['reference'][:split_idx], 3),
        'albedo': (data_files['albedo'][:split_idx], 3),
        'density': (data_files['density'][:split_idx], 1),
        'radiance' : (data_files['radiance_final'][:split_idx], 3),
        'variance' : (data_files['variance_final'][:split_idx], 1),
    }.items() if v[0] is not None}, device=device)
    set.name = "Final"
    training_datasets.append(set)

    validation_datasets = []
    if has_easy:
        set = LitboxDataset({k: v for k, v in {
            'reference': (data_files['reference'][split_idx:], 3),
            'albedo': (data_files['albedo'][split_idx:], 3),
            'density': (data_files['density'][split_idx:], 1),
            'radiance' : (data_files['radiance_easy'][split_idx:], 3),
            'variance' : (data_files['variance_easy'][split_idx:], 1),
        }.items() if v[0] is not None}, device=device)
        set.name = "Easy"
        validation_datasets.append(set)

    if has_medium:
        set =LitboxDataset({k: v for k, v in {
            'reference': (data_files['reference'][split_idx:], 3),
            'albedo': (data_files['albedo'][split_idx:], 3),
            'density': (data_files['density'][split_idx:], 1),
            'radiance' : (data_files['radiance_medium'][split_idx:], 3),
            'variance' : (data_files['variance_medium'][split_idx:], 1),
        }.items() if v[0] is not None}, device=device)
        set.name = "Medium"
        validation_datasets.append(set)
    
    set = LitboxDataset({k: v for k, v in {
        'reference': (data_files['reference'][split_idx:], 3),
        'albedo': (data_files['albedo'][split_idx:], 3),
        'density': (data_files['density'][split_idx:], 1),
        'radiance' : (data_files['radiance_final'][split_idx:], 3),
        'variance' : (data_files['variance_final'][split_idx:], 1),
    }.items() if v[0] is not None}, device=device)
    set.name = "Final"
    validation_datasets.append(set)

    return training_datasets, validation_datasets

def compute_stats(args):
    has_easy = args.input_a_easy_location is not None
    has_medium = args.input_a_medium_location is not None
    cache_dir = get_cache_path(args)
    dataset, loader = create_litbox_raw_data_loader(args)
    sample_count = len(dataset)

    # Compute global stats
    welford_easy = Welford()
    welford_medium = Welford()
    welford_final = Welford()
    welford_density = Welford()
    
    def prepare_for_welford(t: torch.Tensor):
        b, c, _, _ = t.shape
        return t.view(b, c, -1).permute(0, 2, 1).reshape(-1, c)
    
    print("Computing Global Normalization Statistics...")
    print("")
    for index, sample in enumerate(loader):
        if has_easy:
            image_easy, _ = data_processing.preprocess_radiance(sample['input_a_easy'], sample['input_b_easy'], sample['albedo'])
            welford_easy.add_all(prepare_for_welford(image_easy))
        if has_medium:
            image_medium, _ = data_processing.preprocess_radiance(sample['input_a_medium'], sample['input_b_medium'], sample['albedo'])
            welford_medium.add_all(prepare_for_welford(image_medium))
        image_final, _ = data_processing.preprocess_radiance(sample['input_a_final'], sample['input_b_final'], sample['albedo'])
        welford_final.add_all(prepare_for_welford(image_final))
        density = data_processing.preprocess_transmissibility(sample['transmissibility'])
        welford_density.add_all(prepare_for_welford(density))
        print(f"\r{index * g_batch_size:05d} / {sample_count}", end="", flush=True)

    # Compute mean and stddev of all input pixels
    stats = {}
    if has_easy:
        stats['easy_mean'] = welford_easy.mean.tolist()
        stats['easy_stddev'] = torch.sqrt(welford_easy.var_s).tolist()
    if has_medium:
        stats['medium_mean'] = welford_medium.mean.tolist()
        stats['medium_stddev'] = torch.sqrt(welford_medium.var_s).tolist()
    stats['final_mean'] = welford_final.mean.tolist()
    stats['final_stddev'] = torch.sqrt(welford_final.var_s).tolist()
    stats['density_mean'] = welford_density.mean.tolist()
    stats['density_stddev'] = torch.sqrt(welford_density.var_s).tolist()

    # Save stats to a JSON file so they're cached for later.
    stats_file = os.path.join(cache_dir, 'stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=4)

    print()
    pprint.pprint(stats)
    print()
    print(f"Stats saved to {cache_dir}/stats.json.")
    print("-- SUCCESS --")

    return stats

def save_image_batch(batch_index: int, tensor: torch.tensor, template: str):
    for i in range(tensor.shape[0]):
        image_index = batch_index * g_batch_size + i
        save_image(template.format(index=image_index), tensor[i])

def test_batch_already_exists(batch_index: int, template: str):
    for i in range(g_batch_size):
        image_index = batch_index * g_batch_size + i
        if not Path(template.format(index=image_index)).exists():
            return False
    return True

def load_cached_stats(args):
    cache_dir = get_cache_path(args)
    stats_file = os.path.join(cache_dir, 'stats.json')
    if not Path(stats_file).exists():
        print("No cached normalization stats found. They will have to be computed now...")
        return None
    with open(stats_file, 'r') as f:
        return json.load(f)

def build_cache(args):
    has_easy = args.input_a_easy_location is not None
    has_medium = args.input_a_medium_location is not None
    cache_dir = get_cache_path(args)
    dataset, loader = create_litbox_raw_data_loader(args)
    sample_count = len(dataset)

    need_compute_stats = args.recompute_stats

    stats_file = os.path.join(cache_dir, 'stats.json')
    if not Path(stats_file).exists():
        print("No cached normalization stats found. They will have to be computed now...")
        need_compute_stats = True
    
    if need_compute_stats:
        compute_stats(args)

    with open(stats_file, 'r') as f:
        stats = json.load(f)

    if(has_easy):
        easy_mean = torch.tensor(stats['easy_mean'])
        easy_stddev = torch.tensor(stats['easy_stddev'])
    if(has_medium):
        medium_mean = torch.tensor(stats['medium_mean'])
        medium_stddev = torch.tensor(stats['medium_stddev'])
    final_mean = torch.tensor(stats['final_mean'])
    final_stddev = torch.tensor(stats['final_stddev'])
    density_mean = torch.tensor(stats['density_mean'])
    density_stddev = torch.tensor(stats['density_stddev'])

    radiance_easy_template = cache_dir + '/{index:05d}_Radiance_Easy.exr'
    variance_easy_template = cache_dir + '/{index:05d}_Variance_Easy.exr'
    radiance_medium_template = cache_dir + '/{index:05d}_Radiance_Medium.exr'
    variance_medium_template = cache_dir + '/{index:05d}_Variance_Medium.exr'
    radiance_final_template = cache_dir + '/{index:05d}_Radiance_Final.exr'
    variance_final_template = cache_dir + '/{index:05d}_Variance_Final.exr'
    density_template = cache_dir + '/{index:05d}_Density.exr'
    reference_template = cache_dir + '/{index:05d}_Reference.exr'
    albedo_template = cache_dir + '/{index:05d}_Albedo.png'

    print("Preprocessing Images...")
    print("")
    for index, sample in enumerate(loader):
        if has_easy and (not test_batch_already_exists(index, radiance_easy_template) or not test_batch_already_exists(index, variance_easy_template)):
            radiance_easy, variance_easy = data_processing.preprocess_radiance(sample['input_a_easy'], sample['input_b_easy'], sample['albedo'], final_mean, final_stddev)
            save_image_batch(index, radiance_easy, radiance_easy_template)
            save_image_batch(index, variance_easy, variance_easy_template)
        if has_medium and (not test_batch_already_exists(index, radiance_medium_template) or not test_batch_already_exists(index, variance_medium_template)):
            radiance_medium, variance_medium = data_processing.preprocess_radiance(sample['input_a_medium'], sample['input_b_medium'], sample['albedo'], final_mean, final_stddev)
            save_image_batch(index, radiance_medium, radiance_medium_template)
            save_image_batch(index, variance_medium, variance_medium_template)

        if not test_batch_already_exists(index, radiance_final_template) or not test_batch_already_exists(index, variance_final_template):
            radiance_final, variance_final = data_processing.preprocess_radiance(sample['input_a_final'], sample['input_b_final'], sample['albedo'], final_mean, final_stddev)
            save_image_batch(index, radiance_final, radiance_final_template)
            save_image_batch(index, variance_final, variance_final_template)

        if not test_batch_already_exists(index, density_template):
            density = data_processing.preprocess_transmissibility(sample['transmissibility'], density_mean, density_stddev)
            save_image_batch(index, density, density_template)

        if not test_batch_already_exists(index, reference_template):
            reference = data_processing.preprocess_reference(sample['reference'], final_mean, final_stddev)
            save_image_batch(index, reference, reference_template)

        if not test_batch_already_exists(index, albedo_template):
            albedo = data_processing.preprocess_albedo(sample['albedo'])
            save_image_batch(index, albedo, albedo_template)

        print(f"\r{index * g_batch_size:05d} / {sample_count}", end="", flush=True)
    print()
    print("-- SUCCESS --")

def main():
    args = parse_args()
    print(f"Using device: {device}")

    build_cache(args)

if __name__ == "__main__":
    main() 