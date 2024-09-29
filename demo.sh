python main.py --task 'PRE' --dataset 'CIFAR10' --model 'resnet18' --epoch 200
python main.py --task 'OOD' --dataset 'CIFAR10' --OOD 'CIFAR10' --model 'resnet18' --n 1000 --seed_end 1000
python main.py --task 'OOD' --dataset 'CIFAR10' --OOD 'SVHN' --model 'resnet18' --n 1000 --seed_end 1000
python main.py --task 'EO' --dataset 'CIFAR10' --OOD 'SVHN' --model 'resnet18' --n 1000
