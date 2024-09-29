model='resnet18' # 'vgg16' 'mobilenetv2' 'efficientnetv2'
n=1000
num_seed=10000

python main.py --task 'NOISE' --noise 'N' --intensity 0 --model ${model} --n ${n} --seed_end ${num_seed}

for noise in 'G' 'U' 'P' 'S'
do
	for intensity in $(seq 0.0 0.1 1) 
	do
		python main.py --task 'NOISE' --noise ${noise} --intensity ${intensity} --model ${model} --n ${n} --seed_end ${num_seed}
	done
done

for noise in 'G' 'U' 'P' 'S'
do
	for intensity in $(seq 0 0.1 1) 
	do
		python main.py --task 'EN' --noise ${noise} --intensity ${intensity} --model ${model} --n ${n}
	done
done
