# -*- coding:utf-8 -*-
import json
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--plm', type=str, default='bert')
args, unknown = parser.parse_known_args()

# Check if CUDA is available
if torch.cuda.is_available():
    # Get the number of GPU devices
    device = torch.device("cuda:0")

else:
    # If no GPU is available, use CPU
    device = torch.device("cpu")


# datasets = ['physionet']
# datasets = ['P19']
datasets = ['physionet', 'P12', 'P19', 'mimic3']
model_name = args.plm

for dataset in datasets:
	json_file_path = './' + dataset + '/' + dataset + '_variables.json'
	with open(json_file_path, 'r') as json_file:
		data = json.load(json_file)
	path = './plm/' + model_name

	input_text = [value for value in data.values()]

	if model_name == 'bert':
		from transformers import BertTokenizer, BertModel

		# Load BERT model and tokenizer
		model_name = 'bert'
		tokenizer = BertTokenizer.from_pretrained(path)
		model = BertModel.from_pretrained(path).to(device)
		with torch.no_grad():
			# Tokenize and encode
			tokens = tokenizer(input_text, padding=True, truncation=False, return_tensors="pt").to(device)
			outputs = model(**tokens)

			# Get sentence encoding
			sentence_embedding = outputs.last_hidden_state.cpu()[:, 0, :]

	elif model_name == 'bart':
		from transformers import BartTokenizer, BartModel
		tokenizer = BartTokenizer.from_pretrained(path)
		model = BartModel.from_pretrained(path).to(device)
		with torch.no_grad():
			# Tokenize and encode
			tokens = tokenizer(input_text, padding=True, truncation=False, return_tensors="pt").to(device)
			outputs = model(**tokens)

			# Take the last hidden state (average pooling)
			sentence_embedding = outputs.last_hidden_state.mean(dim=1).cpu()

	elif model_name == 'led':
		from transformers import LEDForConditionalGeneration, LEDTokenizer
		sentence_embedding = []
		input_text = [value + " The summary of the above text is" for value in data.values()]
		with torch.no_grad():
			for i in range(len(input_text)):
				tokenizer = LEDTokenizer.from_pretrained(path)
				input_ids = tokenizer(input_text[i], return_tensors="pt").input_ids.to(device)
				global_attention_mask = torch.zeros_like(input_ids)
				global_attention_mask[:, 0] = 1
				model = LEDForConditionalGeneration.from_pretrained(path, return_dict_in_generate=True).to(device)
				outputs = model(input_ids, global_attention_mask=global_attention_mask)
				sentence_embedding.append(outputs.encoder_last_hidden_state.mean(dim=1).cpu())
			sentence_embedding = torch.stack(sentence_embedding, dim=0)[:, 0, :]

	elif model_name == 'gpt2':
		from transformers import GPT2Tokenizer, GPT2Model
		tokenizer = GPT2Tokenizer.from_pretrained(path)
		model = GPT2Model.from_pretrained(path).to(device)
		sentence_embedding = []
		input_text = [value + " The summary of the above text is" for value in data.values()]
		with torch.no_grad():
			for i in range(len(input_text)):
				inputs = tokenizer(input_text[i], return_tensors='pt').to(device)
				outputs = model(**inputs)
				sentence_embedding.append(outputs.last_hidden_state.mean(dim=1).cpu())
			sentence_embedding = torch.stack(sentence_embedding, dim=0)[:, 0, :]

	elif model_name == 'pegasus':
		from transformers import AutoTokenizer, PegasusModel
		tokenizer = AutoTokenizer.from_pretrained(path)
		model = PegasusModel.from_pretrained(path).to(device)
		prompt = [key + '\'s medical properties is: ' for key in data.keys()]
		input_text = [value for value in data.values()]
		with torch.no_grad():
			sentence_embedding = []
			for i in range(len(input_text)):
				input_ids = tokenizer(input_text[i], padding=True, truncation=True, return_tensors="pt").to(
					device).input_ids
				decoder_input_ids = tokenizer(prompt[i], return_tensors="pt").to(device).input_ids
				outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
				sentence_embedding.append(outputs.last_hidden_state.mean(dim=1).cpu())
			sentence_embedding = torch.stack(sentence_embedding, dim=0)[:, 0, :]

	elif model_name == 't5':
		from transformers import T5Tokenizer, T5Model
		with torch.no_grad():
			tokenizer = T5Tokenizer.from_pretrained(path)
			model = T5Model.from_pretrained(path).to(device)
			prompt = [key + '\'s medical properties is: ' for key in data.keys()]
			input_text = [value for value in data.values()]
			sentence_embedding = []
			for i in range(len(input_text)):
				input_ids = tokenizer(input_text[i], padding=True, truncation=True, return_tensors="pt").to(device).input_ids
				decoder_input_ids = tokenizer(prompt[i], return_tensors="pt").to(device).input_ids
				outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
				sentence_embedding.append(outputs.last_hidden_state.mean(dim=1).cpu())
			sentence_embedding = torch.stack(sentence_embedding, dim=0)[:, 0, :]

	# Specify the file path to save
	output_file_path = './' + dataset + '/' + dataset + '_' + model_name + "_var_rep.pt"

	# Save the tensor using PyTorch's torch.save() function
	torch.save(sentence_embedding, output_file_path)