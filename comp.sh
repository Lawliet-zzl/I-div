model='resnet18' # 'vgg16' 'mobilenetv2' 'efficientnetv2'
n=1000
num_seed=10000

python main.py --task 'PRE' --model ${model} --epoch 200


for OOD in 'CIFAR10' 'SVHN' 'CIFAR100' 'STL10' 'DTD' 'F102' 'OP' 'P365' 'Fake' 'SEMEION'
do
	python main.py --task 'OOD' --OOD ${OOD} --model ${model} --n ${n} --seed_end ${num_seed}
done

for OOD in 'SVHN' 'CIFAR100' 'STL10' 'DTD' 'F102' 'OP' 'P365' 'Fake' 'SEMEION'
do
	python main.py --task 'EO' --OOD ${OOD} --model ${model} --n ${n}
done