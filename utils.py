import csv
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import Subset, Dataset, DataLoader
from sklearn.metrics import roc_auc_score

class NoiseInterpolation(object):
	def __init__(self, noise_func, intensity=1.0):
		self.noise_func = noise_func
		self.intensity = intensity
	def __call__(self, tensor):
		noise_img = self.noise_func(tensor)
		return (1 - self.intensity) * tensor + self.intensity * noise_img

def gaussian_noise(tensor, mean=0., std=1.):
	noise = torch.randn(tensor.size()) * std + mean
	return noise

def salt_pepper_noise(tensor, amount=0.05):
	noise = torch.rand(tensor.size())
	salt_pepper_noise = torch.where(noise < amount, torch.ones_like(tensor), torch.zeros_like(tensor))
	return salt_pepper_noise

def uniform_noise(tensor, low=-1, high=1):
	noise = torch.FloatTensor(tensor.size()).uniform_(low, high)
	return noise

def speckle_noise(tensor, mean=0., std=1.):
	noise = torch.randn(tensor.size()) * std + mean
	return tensor * noise

def get_transform(noise, intensity):
	mean = (0.485, 0.456, 0.406)
	std = (0.229, 0.224, 0.225)
	noise_functions = {
	'G': lambda x: gaussian_noise(x, std=0.1),
	'U': lambda x: uniform_noise(x, low=-0.1, high=0.1),
	'S': lambda x: speckle_noise(x, std=0.1),
	'P': lambda x: salt_pepper_noise(x, amount=0.05)}
	if noise == 'N':
		transform = transforms.Compose([
			transforms.Resize((32, 32)),
			transforms.ToTensor(),
			transforms.Normalize(mean, std)
			])
	elif noise in noise_functions:
		transform = transforms.Compose([
			transforms.ToTensor(),
			NoiseInterpolation(noise_functions[noise], intensity=intensity),
			transforms.Normalize(mean, std)
			])
	else:
		raise ValueError('Invalid noise type!')
	return transform

def get_classes(data_name, dataset):

	if data_name == 'CIFAR10':
		data_classes = dataset.classes
	elif data_name == 'CIFAR100':
		data_classes = dataset.classes
	elif data_name == 'STL10':
		data_classes = dataset.classes
	elif data_name == 'SVHN':
		data_classes =  [str(i) for i in range(10)]
	elif data_name == 'OP':
		data_classes = ['Abyssinian', 'American Bulldog', 'American Pit Bull Terrier', 'Basset Hound', 'Beagle', 
		'Bengal', 'Birman', 'Bombay', 'British Shorthair', 'Chihuahua', 'Egyptian Mau', 
		'English Cocker Spaniel', 'English Setter', 'German Shorthaired', 'Great Pyrenees', 
		'Havanese', 'Japanese Chin', 'Keeshond', 'Leonberger', 'Maine Coon', 'Miniature Pinscher', 
		'Newfoundland', 'Persian', 'Pomeranian', 'Pug', 'Ragdoll', 'Russian Blue', 'Saint Bernard', 
		'Samoyed', 'Scottish Terrier', 'Shiba Inu', 'Siamese', 'Sphynx', 'Staffordshire Bull Terrier', 
		'Wheaten Terrier', 'Yorkshire Terrier']
	elif data_name == 'F102':
		data_classes = ['pink primrose','hard-leaved pocket orchid','canterbury bells','sweet pea',
		'english marigold','tiger lily','moon orchid','bird of paradise','monkshood','globe thistle',
		'snapdragon','colt\'s foot','king protea','spear thistle','yellow iris','globe-flower',
		'purple coneflower','peruvian lily','balloon flower','giant white arum lily','fire lily',
		'pincushion flower','fritillary','red ginger','grape hyacinth','corn poppy','prince of wales feathers',
		'stemless gentian','artichoke','sweet william','carnation','garden phlox','love in the mist',
		'mexican aster','alpine sea holly','ruby-lipped cattleya','cape flower','great masterwort',
		'siam tulip','lenten rose','barbeton daisy','daffodil','sword lily','poinsettia','bolero deep blue',
		'wallflower','marigold','buttercup','oxeye daisy','common dandelion','petunia','wild pansy',
		'primula','sunflower','pelargonium','bishop of llandaff','gaura','geranium','orange dahlia',
		'pink-yellow dahlia','cautleya spicata','japanese anemone','black-eyed susan','silverbush',
		'californian poppy','osteospermum','spring crocus','bearded iris','windflower','tree poppy',
		'gazania','azalea','water lily','rose','thorn apple','morning glory','passion flower',
		'lotus','toad lily','anthurium','frangipani','clematis','hibiscus','columbine','desert-rose',
		'tree mallow','magnolia','cyclamen','watercress','canna lily','hippeastrum','bee balm','ball moss',
		'foxglove','bougainvillea','camellia','mallow','mexican petunia','bromelia','blanket flower',
		'trumpet creeper','blackberry lily']
	elif data_name == 'DTD':
		data_classes = dataset.classes
	elif data_name =='C256':
		original_classes = dataset.classes
		data_classes = [name.split('.', 1)[-1] for name in original_classes]
	elif data_name == 'P365':
		data_classes = dataset.classes
	elif data_name == 'Fake':
		data_classes =  ["Class 0", "Class 1"]
	elif data_name == 'SEMEION':
		data_classes =  [str(i) for i in range(10)]
	else:
		data_classes =  None

	# print(data_classes)

	return data_classes

def load_CIFAR10(args):
	mean = (0.485, 0.456, 0.406)
	std = (0.229, 0.224, 0.225)
	transform_train = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(mean, std)
		])
	transform_test = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean, std)
		])
	trainset = datasets.CIFAR10(root=f'{args.root}/CIFAR10', train=True, download=args.download, transform=transform_train)
	testset = datasets.CIFAR10(root=f'{args.root}/CIFAR10', train=False, download=args.download, transform=transform_test)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
	testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
	return trainloader, testloader

def load_ID_data(root, n, batch_size, transform, download):
	dataset = datasets.CIFAR10(root=f'{root}/CIFAR10', train=True, download=download, transform=transform)
	data_classes = dataset.classes
	indices = np.random.choice(len(dataset), n, replace=False)
	dataset = Subset(dataset, indices)
	data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
	return data_loader, data_classes

def load_OOD_data(root, data_name, n, batch_size, transform, download):
	if data_name == 'CIFAR10':
		dataset = datasets.CIFAR10(root=f'{root}/CIFAR10', train=False, download=download, transform=transform)
	elif data_name == 'SVHN':
		dataset = datasets.SVHN(root=f'{root}/SVHN', split='test', download=download, transform=transform)
	elif data_name == 'CIFAR100':
		dataset = datasets.CIFAR100(root=f'{root}/CIFAR100', train=False, download=download, transform=transform)
	elif data_name == 'STL10':
		dataset = datasets.STL10(root=f'{root}/STL10', split='test', download=download, transform=transform)
	elif data_name == 'DTD':
		dataset = datasets.DTD(root=f'{root}/DTD', split='test', download=download, transform=transform)
	elif data_name == 'F102':
		dataset = datasets.Flowers102(root=f'{root}/Flowers102', split='test', download=download, transform=transform)
	elif data_name == 'OP':
		dataset = datasets.OxfordIIITPet(root=f'{root}/OxfordIIITPet', split='test', download=download, transform=transform)
	elif data_name == 'C256':
		dataset = datasets.ImageFolder(root=f'{root}/Caltech256', transform=transform)
	elif data_name == 'P365':
		dataset = datasets.Places365(root=f'{root}/Places365', split='val', small =True, download=download, transform=transform)
	elif data_name == 'Fake':
		dataset = datasets.FakeData(size=n, image_size=(3, 32, 32), transform=transform)
	elif data_name == 'SEMEION':
		dataset = datasets.SEMEION(root=f'{root}/SEMEION', download=download, transform=transforms.Compose([transforms.Grayscale(num_output_channels=3), *transform.transforms]))

	data_classes = get_classes(data_name, dataset)
	indices = np.random.choice(len(dataset), n, replace=False)
	dataset = Subset(dataset, indices)
	data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
	return data_loader, data_classes

def load_IDOOD_data(args):
	transform = get_transform('N', 0)
	ID_loader, ID_classes = load_ID_data(args.root, args.n, args.batch_size, transform, args.download)
	OOD_loader, OOD_classes = load_OOD_data(args.root, args.OOD, args.n, args.batch_size, transform, args.download)
	return ID_loader, ID_classes, OOD_loader, OOD_classes

def load_noise_data(args):
	transform_clean = get_transform('N', 0)
	transform_noise = get_transform(args.noise, args.intensity)
	ID_loader, _ = load_ID_data(args.root, args.n, args.batch_size, transform_clean, args.download)
	OOD_loader, _ = load_OOD_data(args.root, args.dataset, args.n, args.batch_size, transform_noise, args.download)
	return ID_loader, OOD_loader

def build_model(model, num_classes):
	if model == 'resnet18':
		net = models.resnet18()
		net.fc = nn.Linear(net.fc.in_features, num_classes)
	elif model == 'vgg16':
		net = models.vgg16()
		net.classifier[6] = nn.Linear(net.classifier[6].in_features, num_classes)
	elif model == 'densenet121':
		net = models.densenet121()
		net.classifier = nn.Linear(net.classifier.in_features, num_classes)
	elif model == 'mobilenetv2':
		net = models.mobilenet_v2()
		net.classifier[1] = nn.Linear(net.classifier[1].in_features, num_classes)
	elif model == 'efficientnetv2':
		net = models.efficientnet_v2_s()
		net.classifier[1] = nn.Linear(net.classifier[1].in_features, num_classes)

	# if torch.cuda.device_count() > 1:
	# 	print("Let's use", torch.cuda.device_count(), "GPUs!")
	# 	net = nn.DataParallel(net)
	net = net.cuda()
	return net

def cal_auroc(ID_scores, OOD_scores):
	true_labels = [0] * len(ID_scores) + [1] * len(OOD_scores)
	scores_combined = ID_scores + OOD_scores
	auroc = roc_auc_score(true_labels, scores_combined)
	return auroc*100

def save_result(log_name, info):
	with open(log_name, 'a') as logfile:
		logwriter = csv.writer(logfile, delimiter=',')
		logwriter.writerow(info)


def save_model(model, pth_name):
	torch.save(model.state_dict(), pth_name)
	
def read_csv(file_name):
	list_acc = []
	list_dis = []
	with open(file_name, 'r') as file:
		csv_reader = csv.reader(file)
		next(csv_reader)
		for row in csv_reader:
			list_acc.append(float(row[1]))
			list_dis.append(float(row[2]))
	mean_acc = sum(list_acc) / len(list_acc) if list_acc else 0
	return mean_acc*100, list_dis