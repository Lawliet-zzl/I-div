# Code for "Revealing Distribution Discrepancy by Sampling Transfer in Unlabeled Data"

## Environment Setup

To install all dependencies, use the following command to create the environment from the provided `yaml` file:

```bash
conda env create -f pynet_environment.yaml
```

## Data Preparation

By default, the data is stored in the `./data` folder. If you prefer to store the data in a different folder, you can modify the path in the code.

To automatically download the dataset, add the `--download` argument when running the code. For example:

```python
python main.py --task 'PRE' --dataset 'CIFAR10' --model 'resnet18' --epoch 200 --download
```

## Usage Examples

Here are the main features of the project with corresponding command-line examples:

1. **Pre-train the network**:
   
   Pre-train a ResNet18 model on the `CIFAR10` dataset:

   ```bash
   python main.py --task 'PRE' --dataset 'CIFAR10' --model 'resnet18' --epoch 200
   ```

2. **Calculate the distribution discrepancy between CIFAR10 and CIFAR10**:

   ```bash
   python main.py --task 'OOD' --dataset 'CIFAR10' --OOD 'CIFAR10' --model 'resnet18' --n 1000 --seed_end 1000
   ```

3. **Calculate the distribution discrepancy between CIFAR10 and SVHN**:

   ```bash
   python main.py --task 'OOD' --dataset 'CIFAR10' --OOD 'SVHN' --model 'resnet18' --n 1000 --seed_end 1000
   ```

4. **Calculate AUROC and evaluate classification**:

   ```bash
   python main.py --task 'EO' --dataset 'CIFAR10' --OOD 'SVHN' --model 'resnet18' --n 1000
   ```

## Script Descriptions

Run the following script to compute distances between all dataset pairs:
```bash
sh comp.sh
```

Run the following script to compute distances between clean data and data with different noise levels:
```bash
sh noise.sh
```
