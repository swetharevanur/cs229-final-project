import argparse
import os, sys
import time
import numpy as np
import torch
import torch.nn as nn
import torchtext.vocab as vocab
from torchtext.data import Iterator, BucketIterator
import copy
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.test.utils import common_texts, get_tmpfile
from rnn_classifier.model import *
torch.manual_seed(7)

sys.stdout = open(os.path.join('output', 'logs', 'lstm.log'), 'a+')

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

NUM_LABELS = 6

def update_stats(accuracy, confusion_matrix, outputs, y):
	_, max_ind_out = torch.max(outputs, 1)
	_, max_ind_y = torch.max(y, 1)
	equal = torch.eq(max_ind_out, max_ind_y)
	correct = int(torch.sum(equal))

	for j, i in zip(max_ind_out, max_ind_y):
		confusion_matrix[int(i), int(j)] += 1

	return accuracy + correct, confusion_matrix

def train_model(model, train_batcher, val_batcher, optimizer, criterion, dataset_sizes, hyperparams):
	model.train()

	since = time.time()

	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0
	
	train_loss_history = []
	train_acc_history, val_acc_history = [], []

	dataloaders = {'train': train_batcher, 'val': val_batcher}

	num_epochs = hyperparams['num_epochs']
	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)
		sys.stdout.flush()

		# Each epoch has a training and validation phase
		for phase in ['train', 'val']:
			accuracy, confusion_matrix = 0, np.zeros((NUM_LABELS, NUM_LABELS), dtype = int)
			if phase == 'train': model.train()  # Set model to training mode
			else: model.eval()   # Set model to evaluate mode

			running_loss = 0.0
			running_corrects = 0

			# Iterate over data
			for inputs, labels in dataloaders[phase]:
				inputs = inputs.to(device)
				labels = labels.to(device)

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward
				# track history if only in train
				with torch.set_grad_enabled(phase == 'train'):
					outputs, _ = model(inputs)
					loss = criterion(outputs, labels)

					if phase == 'train':
						train_loss_history.append(loss) # per batch loss
						loss.backward()
						torch.nn.utils.clip_grad_norm_(model.parameters(), hyperparams['grad_clipping'])
						optimizer.step()

				# statistics
				running_loss += loss.item() * inputs.size(0)
				# running_corrects += torch.sum(preds == labels.data)
				accuracy, confusion_matrix = update_stats(accuracy, confusion_matrix, outputs, labels)

			epoch_loss = running_loss/dataset_sizes[phase]
			epoch_acc = accuracy/dataset_sizes[phase]
				
			if phase == 'train':
				train_acc_history.append(float(epoch_acc)) # per epoch
			else:
				val_acc_history.append(float(epoch_acc)) # per epoch
				
				
			print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
			print(confusion_matrix)
			sys.stdout.flush()

			# deep copy the model
			if phase == 'val' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())

		print()

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))
	print('Best val acc: {:4f}'.format(best_acc))
	sys.stdout.flush()

	# load best model weights
	model.load_state_dict(best_model_wts)
	return model, best_acc, train_loss_history, train_acc_history, val_acc_history

class BatchWrapper:
	def __init__(self, dl, x_var, y_vars, device = device):
		self.dl, self.x_var, self.y_vars = dl, x_var, y_vars # we pass in the list of attributes for x

	def __iter__(self):
		for batch in self.dl:
			x = getattr(batch, self.x_var) # we assume only one input in this wrapper
			if self.y_vars is not None:
				y = torch.cat([getattr(batch, feat).unsqueeze(1) for feat in self.y_vars], dim = 1).float()
			else:
				y = torch.zeros((1))
			
			yield (x, y)
  
	def __len__(self):
		return len(self.dl)

def train_handler(hyperparams, text_datasets, TEXT, LABEL, num_tokens):
	train, val = text_datasets['train'], text_datasets['val'] 

	splits = ['train', 'val']
	# dataloaders = {x: torch.utils.data.DataLoader(text_datasets[x], hyperparams['batch_size'], 
	# 	shuffle = True, num_workers = 16) for x in splits}
	dataset_sizes = {x: len(text_datasets[x]) for x in splits}
	print(dataset_sizes)
	sys.stdout.flush()

	train_iter, val_iter = BucketIterator.splits((train, val), 
		batch_sizes = (hyperparams['batch_size'], hyperparams['batch_size']),
		device = -1,
		sort_key = lambda x: len(x.text),
		sort_within_batch = False,
		repeat = False)

	train_batcher = BatchWrapper(train_iter, 'text',\
		['damaged_infrastructure', 'damaged_nature', 'fires', 'flood', 'human_damage', 'non_damage'], device = device)
	val_batcher = BatchWrapper(val_iter, 'text', \
		['damaged_infrastructure', 'damaged_nature', 'fires', 'flood', 'human_damage', 'non_damage'], device = device)

	# set up randomly-initialized embedding layer 
	embedding = nn.Embedding(num_tokens, 100, padding_idx = 1, max_norm = 1)
	
	# set up RNN encoder
	encoder = Encoder(hyperparams['embedding_size'], 
					  hyperparams['num_hidden_units'], 
					  nlayers = hyperparams['num_layers'], 
					  dropout = hyperparams['dropout_rate'], 
					  bidirectional = hyperparams['is_bidirectional'], 
					  rnn_type = hyperparams['rnn_type'])

	# set up self-attention
	attention_dim = hyperparams['num_hidden_units'] if not hyperparams['is_bidirectional'] \
													else 2 * hyperparams['num_hidden_units']
	attention = Attention(attention_dim, attention_dim, attention_dim)

	# set up FC layers
	model = Classifier(embedding, encoder, attention, attention_dim, NUM_LABELS)
	model.to(device)

	criterion = nn.L1Loss()
	optimizer = torch.optim.Adam(model.parameters(), hyperparams['init_lr'], amsgrad = True)


	return train_model(model, train_batcher, val_batcher, optimizer, 
		criterion, dataset_sizes, hyperparams)
	
def test_handler(model, hyperparams, text_datasets):
	# dataloaders = {x: torch.utils.data.DataLoader(text_datasets[x], hyperparams[batch_size], 
	# 	shuffle = True, num_workers = 16) for x in splits}
	dataset_sizes = {x: len(text_datasets[x]) for x in ['test']}

	test_iter = Iterator(test, 
		batch_size = hyperparams['batch_size'], 
		device = -1,
		sort = False, 
		sort_within_batch = False, 
		repeat = False)
	test__batcher = BatchWrapper(test_iter, 'text', None, device = device)

	
	preds_list, gt_list = [], []
	
	model.eval()
	with torch.no_grad():
		for inputs, labels in dataloaders['test']:
			inputs = inputs.to(device)
			labels = labels.to(device)
			outputs = model(inputs)
			preds_list.extend(list(np.asarray(torch.max(outputs, 1)[1])))
			gt_list.extend(list(np.asarray(labels)))
	
	return preds_list, gt_list