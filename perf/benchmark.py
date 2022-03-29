import time
import torch
import torchani
import pynvml
import gc
import os
from ase.io import read
import argparse
import pandas


runcounter = 0
N = 10
df = []


def getGpuName(device=None):
    i = device if device else torch.cuda.current_device()
    real_i = int(os.environ['CUDA_VISIBLE_DEVICES'][0]) if 'CUDA_VISIBLE_DEVICES' in os.environ else i
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(real_i)
    name = pynvml.nvmlDeviceGetName(h)
    return name.decode("utf-8")


def checkgpu(device=None):
    i = device if device else torch.cuda.current_device()
    t = torch.cuda.get_device_properties(i).total_memory
    c = torch.cuda.memory_reserved(i)
    name = torch.cuda.get_device_properties(i).name
    print('   GPU Memory Cached (pytorch) : {:7.1f}MB / {:.1f}MB ({})'.format(c / 1024 / 1024, t / 1024 / 1024, name))
    real_i = int(os.environ['CUDA_VISIBLE_DEVICES'][0]) if 'CUDA_VISIBLE_DEVICES' in os.environ else i
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(real_i)
    info = pynvml.nvmlDeviceGetMemoryInfo(h)
    name = pynvml.nvmlDeviceGetName(h)
    print('   GPU Memory Used (nvidia-smi): {:7.1f}MB / {:.1f}MB ({})'.format(info.used / 1024 / 1024, info.total / 1024 / 1024, name.decode()))
    return f'{(info.used / 1024 / 1024):.1f}MB'

def format_time(t):
    if t < 1:
        t = f'{t * 1000:.1f} ms'
    else:
        t = f'{t:.3f} sec'
    return t

def benchmark(speciesPositions, aev_comp, runbackward=False, mol_info=None, verbose=True):
    global runcounter

    runname = f"{'cu' if aev_comp.use_cuda_extension else 'py'} aev fd{'+bd' if runbackward else''}"

    forward_time = 0
    force_time = 0
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    start = time.time()

    gpumem = None
    for i in range(N):
        species, coordinates = speciesPositions
        coordinates = coordinates.requires_grad_(runbackward)

        torch.cuda.synchronize()
        forward_start = time.time()

        # cell list uses matmul, we should disable TF32 to get accurate result
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.nvtx.range_push('AEV forward')
        _, aev = aev_comp((species, coordinates))
        torch.cuda.nvtx.range_pop()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if args.mode == 'single':
            torch.cuda.nvtx.range_push('NN forward')
            species_energies = single_model((species, aev))
            torch.cuda.nvtx.range_pop()
        elif args.mode == 'ensemble':
            torch.cuda.nvtx.range_push('NN forward')
            species_energies = neural_networks((species, aev))
            torch.cuda.nvtx.range_pop()
        elif args.mode == 'aev':
            species_energies = (species, aev.sum(dim=-1))
        else:
            raise ValueError("Illegal argument")
        energies = species_energies[1]

        torch.cuda.synchronize()
        forward_time += time.time() - forward_start

        if runbackward:  # backward
            force_start = time.time()
            torch.cuda.nvtx.range_push('Backward')
            force = -torch.autograd.grad(energies.sum(), coordinates, create_graph=True, retain_graph=True)[0]
            torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize()
            force_time += time.time() - force_start

        if i == 2 and verbose:
            gpumem = checkgpu()

    torch.cuda.synchronize()
    total_time = (time.time() - start) / N
    force_time = force_time / N
    forward_time = forward_time / N
    others_time = total_time - force_time - forward_time

    if verbose:
        print(f'  Duration: {total_time * N:.2f} s')
        print(f'  Speed: {total_time*1000:.2f} ms/it')
        df.append({
            'RUN': f'{(runcounter+1):02} {runname}',
            'PDB': f"{mol_info['name']}",
            'Size': f"{mol_info['atoms']}",
            'Forward': f'{format_time(forward_time)}',
            'Backward': f'{format_time(force_time)}',
            'Others': f'{format_time(others_time)}',
            'Total': f'{format_time(total_time)}',
            'GPU': f'{gpumem}'
        })
        runcounter += 1


def run(file, nnp_ref, runbackward, maxatoms=10000):
    species = torch.tensor([mol.get_atomic_numbers()], device=device)
    positions = torch.tensor([mol.get_positions()], dtype=torch.float32, requires_grad=False, device=device)
    spelist = list(torch.unique(species.flatten()).cpu().numpy())
    realmolsize = species.shape[-1]
    species = species[:, :maxatoms]
    positions = positions[:, :maxatoms, :]
    molsize = species.shape[-1]
    speciesPositions = nnp_ref.species_converter((species, positions))
    print(f'File: {file}, Molecule size: {molsize} / {realmolsize}, Species: {spelist}\n')

    mol_info = {'name': file, 'atoms': species.shape[-1]}
    # warm up
    benchmark(speciesPositions, nnp_ref.aev_computer, runbackward, mol_info, verbose=False)

    # benchmark
    benchmark(speciesPositions, nnp_ref.aev_computer, runbackward, mol_info)
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--system_size', type=int)
    parser.add_argument('--mode',
                        type=str,
                        help='Benchmark mode, can be: aev(only run AEV), single(use a single network), ensemble(use an ensemble of networks)',
                        default=False)
    args = parser.parse_args()

    device = torch.device('cuda')
    nnp_ref = torchani.models.ANI2x(periodic_table_index=True, model_index=None, cell_list=True,
                                    use_cuaev_interface=True, use_cuda_extension=True).to(device)

    neural_networks = nnp_ref.neural_networks
    single_model = neural_networks[0]
    neural_networks = neural_networks.to_infer_model(use_mnp=True).to(device)
    single_model = single_model.to_infer_model(use_mnp=True).to(device)

    file = 'water-20.pdb'
    mol = read(file)
    # forward only
    # run(file, nnp_ref, runbackward=False, maxatoms=args.system_size)
    # with backward
    run(file, nnp_ref, runbackward=True, maxatoms=args.system_size)

    print(pandas.DataFrame(df))
