import argparse
import os
import numpy as np
import torch
import clip

from tqdm import tqdm
from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=1000)
parser.add_argument('--seed_start', type=int, default=0)
parser.add_argument('--seed_end', type=int, default= 1)
parser.add_argument('--num_classes', default=10, type=int, help='number of Classes')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--dataset', default="CIFAR10", type=str, help='dataset')
parser.add_argument('--model', default="resnet18", type=str, help='model type (default: ResNet18)')
parser.add_argument('--decay', default=5e-4, type=float, help='weight decay 5e-4')
parser.add_argument('--OOD', default="SVHN", type=str)
parser.add_argument('--lambd', type=float, default=100.0)
parser.add_argument('--gamma', type=float, default=1.0)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--intensity', default=0.5, type=float, help='[0, 0.05, 0.5]')
parser.add_argument('--noise', default="G", type=str, help='N, G, U, P, S')
parser.add_argument('--root', default="./data", type=str)
parser.add_argument('--task', type=str, default="OOD", help='PRE, OOD, NOISE, EVAL')
parser.add_argument('--download', action='store_true', help="Set this flag to download data")
args = parser.parse_args()

# "/mnt/Datasets"

class DRELoss(nn.Module):
	def __init__(self, lambd):
		super(DRELoss, self).__init__()
		self.lambd = lambd
	def forward(self, r, t):
		loss_KL = (t * torch.log(r)).mean()
		loss_cons = torch.norm((torch.pow(r, t)).mean() - 1, p=2)
		loss = loss_KL + self.lambd * loss_cons
		return loss

class IDiv(nn.Module):
	def __init__(self):
		super(IDiv, self).__init__()
	def forward(self, outputs, labels, r, v):
		log_softmax_outputs = F.log_softmax(outputs, dim=1)
		gathered_outputs = -log_softmax_outputs.gather(1, labels.unsqueeze(1)).squeeze(1)
		weight = torch.abs(r*v - 1)
		weighted_losses = gathered_outputs*weight
		dicrepancy = weighted_losses.sum()
		return dicrepancy.item()

class Adapter(nn.Module):
	def __init__(self, pretrained_model, in_features = 10):
		super(Adapter, self).__init__()
		self.pretrained_model = pretrained_model
		self.downscale = nn.Linear(in_features, 64)
		self.upscale = nn.Linear(64, in_features)
		self.GELU = nn.GELU()
		self.Softplus = nn.Softplus()
	def forward(self, x):
		with torch.no_grad():
			y = self.pretrained_model(x)
		out = self.downscale(y)
		out = self.GELU(out)
		out = self.upscale(out)
		out = self.GELU(out)
		out = torch.mean(out, dim=1)
		out = self.Softplus(out)
		return out

def init_setting():
	if not os.path.isdir('results'):
		os.mkdir('results')
	if not os.path.isdir('data'):
		os.mkdir('data')
	if not os.path.isdir('checkpoints'):
		os.mkdir('checkpoints')
	pth_name = f'./checkpoints/{args.dataset}_{args.model}.pth'

	if args.task == 'PRE':
		return pth_name, None
	else:
		if args.task == 'OOD' or args.task == 'EO':
			res_name = ('results/' + args.dataset + '_' + args.model + '_' + args.OOD + '_' + str(args.n) + '.csv')
		if args.task == 'NOISE' or args.task == 'EN':
			res_name = ('results/' + args.dataset + '_' + args.model + '_' + args.noise + '_' + str(args.intensity) + '_' + str(args.n) + '.csv')
		if not os.path.exists(res_name):
			with open(res_name, 'w') as logfile:
				logwriter = csv.writer(logfile, delimiter=',')
				logwriter.writerow(['Seed', 'accuracy', 'discrepancy'])
		return pth_name, res_name

def get_embeddings(clip_model, data_class):
	text_prompts = [f"A photo of a {label}" for label in data_class]
	text_inputs = clip.tokenize(text_prompts).cuda()
	with torch.no_grad():
		text_features = clip_model.encode_text(text_inputs)
	text_features /= text_features.norm(dim=-1, keepdim=True)
	return text_features 

def get_label_map(ID_classes, OOD_classes):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	clip_model, preprocess = clip.load("ViT-B/32", device=device)
	# clip_model, preprocess = clip.load("ViT-B/16", device=device)
	ID_features = get_embeddings(clip_model, ID_classes)
	OOD_features = get_embeddings(clip_model, OOD_classes)
	similarity_matrix = ID_features @ OOD_features.T
	return similarity_matrix

def test_with_CLIP(model, test_loader, similarity_matrix):
	model.eval()
	corrects = 0
	total = 0
	similarity_matrix = similarity_matrix.float()

	with torch.no_grad():
		for i, (inputs, labels) in enumerate(test_loader):

			inputs, labels = inputs.cuda(), labels.cuda()
			outputs = model(inputs)

			probabilities = nn.functional.softmax(outputs, dim=1)
			mapped_features = torch.matmul(probabilities, similarity_matrix)
			probabilities = nn.functional.softmax(mapped_features, dim=1)
			predicted = torch.argmax(probabilities, dim=1)

			corrects += torch.sum(predicted == labels.data)
			total += labels.size(0)

	accuracy = corrects.double() / total * 100
	# print(f'Test Accuracy with {args.model} and CLIP is : {accuracy:.2f}% on {args.OOD}')	
	return accuracy.item()

def evaluate(model, adapter, ID_loader):
	model.eval()
	adapter.eval()
	measure = IDiv()
	criterion = nn.CrossEntropyLoss()
	dicrepancy = 0.0
	num_ID = 0

	EV = 0.0
	gamma = args.gamma
	target_min, target_max = 0.5, 1.5

	with torch.no_grad():
		for data_ID, labels_ID in ID_loader:
			data_ID, labels_ID = data_ID.cuda(), labels_ID.cuda()
			num_ID += data_ID.size(0)
			r_ID = adapter(data_ID)
			transformed_r_ID = (gamma * r_ID * r_ID) / (gamma * r_ID - 1)
			EV += transformed_r_ID.sum().item()

		EV = EV / num_ID

		min_v_ID = float('inf')
		max_v_ID = float('-inf')

		for data_ID, labels_ID in ID_loader:
			data_ID, labels_ID = data_ID.cuda(), labels_ID.cuda()
			r_ID = adapter(data_ID)
			v_ID = (gamma * r_ID - 1) / (gamma * r_ID * r_ID) * EV
			min_v_ID = min(min_v_ID, v_ID.min().item())
			max_v_ID = max(max_v_ID, v_ID.max().item())

		for data_ID, labels_ID in ID_loader:
			data_ID, labels_ID = data_ID.cuda(), labels_ID.cuda()
			r_ID = adapter(data_ID)
			v_ID = (gamma * r_ID - 1) / (gamma * r_ID * r_ID) * EV

			v_ID_normalized = target_min + ((v_ID - min_v_ID) / (max_v_ID - min_v_ID)) * (target_max - target_min)
			dicrepancy += measure(model(data_ID), labels_ID, r_ID, v_ID_normalized)
			dicrepancy = dicrepancy / num_ID

		# print(f"v_ID for {args.OOD}: {min_v_ID} - {max_v_ID}")
	return dicrepancy

def train(model, train_loader):
	model.train()
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)
	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
	for epoch in range(args.epoch):
		running_loss = 0.0
		num = 0
		for images, labels in train_loader:
			images, labels = images.cuda(), labels.cuda()
			num += images.size(0)
			optimizer.zero_grad()
			outputs = model(images)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
		scheduler.step()
		running_loss = running_loss/len(train_loader)
		if (epoch + 1) % 10 == 0:
			print(f"[{args.dataset}, {args.model}] Epoch {epoch+1}/{args.epoch} - Loss: {running_loss}")

def test(model, test_loader):
	model.eval()
	criterion = nn.CrossEntropyLoss()
	total = 0
	correct = 0
	with torch.no_grad():
		for images, labels in test_loader:
			images, labels = images.cuda(), labels.cuda()
			outputs = model(images)
			loss = criterion(outputs, labels)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
		test_acc = 100 * correct / total
	return test_acc

def train_adapter(model, adapter, ID_loader, OOD_loader):
	adapter.train()
	criterion = DRELoss(args.lambd)
	optimizer = optim.Adam(adapter.parameters(), lr=0.001, weight_decay=0.0001)

	for epoch in range(args.epoch):
		for (data_ID, _), (data_OOD, _) in zip(ID_loader, OOD_loader):
			data_ID, data_OOD = data_ID.cuda(), data_OOD.cuda()

			target_combined = torch.cat([
				torch.ones(data_ID.size(0), dtype=torch.float32).cuda(),
				torch.full((data_OOD.size(0),), -1, dtype=torch.float32).cuda()
				], dim=0)

			data_combined = torch.cat((data_ID, data_OOD), dim=0)
			r = adapter(data_combined)
			loss = criterion(r, target_combined)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

def est_discrepany(model, num_classes, ID_loader, OOD_loader):
	adapter = Adapter(model, in_features = num_classes).cuda()
	train_adapter(model, adapter, ID_loader, OOD_loader)
	discrepancy = evaluate(model, adapter, ID_loader)
	return discrepancy

def main():

	pth_name, res_name = init_setting()
	model = build_model(args.model, args.num_classes)

	if args.task == 'PRE':
		train_loader, test_loader = load_CIFAR10(args)
		model = build_model(args.model, args.num_classes)
		print(f'Training {args.model} on {args.dataset}')
		train(model, train_loader)
		test_acc = test(model, test_loader)
		print(f"Test accuracy: {test_acc}")
		save_model(model, pth_name)

	if args.task == 'OOD':
		model.load_state_dict(torch.load(pth_name))
		for seed in range(args.seed_start, args.seed_end):
			torch.manual_seed(seed)
			ID_loader, ID_classes, OOD_loader, OOD_classes = load_IDOOD_data(args)
			similarity_matrix = get_label_map(ID_classes, OOD_classes)
			discrepancy = est_discrepany(model, args.num_classes, ID_loader, OOD_loader)
			accuracy = test_with_CLIP(model, OOD_loader, similarity_matrix)
			save_result(res_name, [seed, accuracy, discrepancy])
			print(f'[{seed}, {args.dataset}, {args.OOD}, {args.model}]: discrepancy: {discrepancy:.3f}, accuracy: {accuracy:.1f}%')
	
	if args.task == 'NOISE':
		model.load_state_dict(torch.load(pth_name))
		for seed in range(args.seed_start, args.seed_end):
			ID_loader, OOD_loader = load_noise_data(args)
			discrepancy = est_discrepany(model, args.num_classes, ID_loader, OOD_loader)
			accuracy = test(model, OOD_loader)
			save_result(res_name, [seed, accuracy, discrepancy])
			print(f'[{seed}, {args.dataset} with {args.noise}({args.intensity}), {args.model}]: discrepancy: {discrepancy:.3f}, accuracy: {accuracy:.1f}%')

	if args.task == 'EO':
		ID_name = ('results/' + args.dataset + '_' + args.model + '_' + args.dataset + '_' + str(args.n) + '.csv')
		ID_mean_acc, ID_list_dis = read_csv(ID_name)
		OOD_mean_acc, OOD_list_dis = read_csv(res_name)
		auroc = cal_auroc(ID_list_dis, OOD_list_dis)
		print(f'[{args.dataset} VS {args.OOD}, {args.model}] - AUORC: {auroc:.2f}, ACC: {OOD_mean_acc:.1f}')

	if args.task == 'EN':
		ID_name = ('results/' + args.dataset + '_' + args.model + '_N_0.0_' + str(args.n) + '.csv')
		ID_mean_acc, ID_list_dis = read_csv(ID_name)
		OOD_mean_acc, OOD_list_dis = read_csv(res_name)
		auroc = cal_auroc(ID_list_dis, OOD_list_dis)
		print(f'[{args.dataset} VS {args.noise}({args.intensity}, {args.model}] - AUORC: {auroc:.2f}, ACC: {OOD_mean_acc:.1f}')

if __name__ == "__main__":
	main()