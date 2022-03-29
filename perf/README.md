# Prepare the environment

## Step 1: Environment setup

We will use anaconda for our environment.
To begin with, we create and activate a new conda environment called `ani1`:

```bash
conda create --name ani1 python==3.8
conda activate ani1
```

After having this envirionment, we should install some necessary packages:
```bash
conda install -c pytorch torchvision=0.11.3 pytorch=1.10.2 cudatoolkit=11.3
conda install -c conda-forge openmm=7.6 openmmtools=0.21.2 pynvml=11.4.1 ase=3.22.1 h5py=3.6.0
```

## Step 2: Generate the input system

We need to generate a water box as an input:
```bash
python waterbox.py
```
This can take a few minutes.
After finish, you will get a `water-20.pdb` file.

## Step 3: Install TorchANI

Our group's code is TorchANI.
Currently, the latest TorchANI is hosted in a private repository of the Roitberg group.
https://github.com/roitberg-group/torchani_sandbox

To install TorchANI, we need first clone:

```bash
git clone git@github.com:roitberg-group/torchani_sandbox.git
```

Then install

```bash
cd torchani_sandbox
git checkout withnbrlist
python setup.py develop --ext
cd ..
```

Note that this project needs to use the `withnbrlist` branch.

# Run the experiment

You can use `benchmark.py` to get the time and memory usage of different benchmark mode and different system sizes.

## 400k system

To benchmark the AEV calculation performance, use
```
python benchmark.py --mode aev --system_size 400000
```

Which generates the following output on NVIDIA A100-SXM4-80GB:

```
               RUN           PDB    Size  Forward Backward  Others    Total        GPU
0     01 cu aev fd  water-20.pdb  400000  55.9 ms   0.0 ms  2.3 ms  58.2 ms  13546.0MB
1  02 cu aev fd+bd  water-20.pdb  400000  55.8 ms  23.6 ms  0.1 ms  79.5 ms  14602.0MB
```

To benchmark the single network inference performance, use
```
python benchmark.py --mode single --system_size 400000
```

Which generates the following output on NVIDIA A100-SXM4-80GB:

```
               RUN           PDB    Size  Forward Backward  Others     Total        GPU
0     01 cu aev fd  water-20.pdb  400000  63.5 ms   0.0 ms  1.7 ms   65.2 ms  16420.0MB
1  02 cu aev fd+bd  water-20.pdb  400000  86.4 ms  37.6 ms  0.1 ms  124.1 ms  20792.0MB
```

To benchmark the network ensemble inference performance, use
```
python benchmark.py --mode ensemble --system_size 400000
```

Which generates the following output on NVIDIA A100-SXM4-80GB:

```
               RUN           PDB    Size   Forward  Backward  Others     Total        GPU
0     01 cu aev fd  water-20.pdb  400000  108.3 ms    0.0 ms  1.8 ms  110.2 ms  28540.0MB
1  02 cu aev fd+bd  water-20.pdb  400000  126.6 ms  267.1 ms  0.1 ms  393.8 ms  68924.0MB
```
