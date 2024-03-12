import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import evaluate
from torch.utils.tensorboard import SummaryWriter
################################################
##       Part2 --- Language Modeling          ##
################################################   
class VanillaLSTM(nn.Module):
    def __init__(self, vocab_size,
                 embedding_dim,
                 hidden_dim,
                 num_layers,
                 dropout_rate,
                 embedding_weights=None,
                 freeze_embeddings=False):
                
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # pass embeeding weights if exist
        if embedding_weights is not None:
            self.embedding = torch.nn.Embedding.from_pretrained(torch.from_numpy(embedding_weights).float())
            self.embedding.weight.requires_grad = not freeze_embeddings
        else:  # train from scratch embeddings
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Define bi-directional LSTM layer with `num_layers` and `dropout_rate`.
        # The input dimension is the embedding dimension and the output dimension is the hidden dimension.
        # Set bidirectional to True to make the LSTM bi-directional.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout_rate, bidirectional=True, batch_first=True)
        
        self.dropout = nn.Dropout(dropout_rate)
        # Define the feedforward layer with `num_layers` and `dropout_rate`.
        # The input dimension is 2 times the hidden dimension (because of the bi-directional LSTM) and the output dimension is the vocab size.
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, input_id):
        embedding = self.dropout(self.embedding(input_id))
        
        # Get output from (LSTM layer-->dropout layer-->feedforward layer)
        ## You can add several lines of code here.
        #print("VanillaLSTM: embedding:", embedding.size())
        output, hidden = self.lstm(embedding)
        #print("VanillaLSTM: hidden:", hidden[0].size(), hidden[1].size())
        #print("VanillaLSTM: output:", output.size())
        output = self.dropout(output)
        output = self.fc(output)
        #print("VanillaLSTM: output:", output.size())
        return output

def train_lstm(model, train_loader, optimizer, criterion, device="cuda:0", tensorboard_path="./tensorboard"):
    """
    Main training pipeline. Implement the following:
    - pass inputs to the model
    - compute loss
    - perform forward/backward pass.
    """
    tb_writer = SummaryWriter(tensorboard_path)
    # Training loop
    model.train()
    running_loss = 0
    epoch_loss = 0
    for i, data in enumerate(tqdm(train_loader)):
        # get the inputs
        inputs = data.to(device)
        
        # TODO: get the language modelling labels form inputs
        labels = data[:, 1:].to(device)
        
        # TODO: Implement forward pass. Compute predicted y by passing x to the model
        y_pred = model(inputs)
        y_pred = y_pred[:, :-1, :].permute(0, 2, 1)

        # TODO: Compute loss
        loss = criterion(y_pred, labels)
        
        # TODO: Implement Backward pass. 
        # Hint: remember to zero gradients after each update. 
        # You can add several lines of code here.
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        running_loss += loss.item()
        epoch_loss += loss.item()
        if i>0 and i % 500 == 0. :
            print(f'[Step {i + 1:5d}] loss: {running_loss / 500:.3f}')
            tb_writer.add_scalar("lstm/train/loss", running_loss / 500, i)
            running_loss = 0.0

    tb_writer.flush()
    tb_writer.close()
    print(f'Epoch Loss: {(epoch_loss / len(train_loader)):.4f}')
    return epoch_loss / len(train_loader)

def test_lstm(model, test_loader, criterion, device="cuda:0"):
    """
    Main testing pipeline. Implement the following:
    - get model predictions
    - compute loss
    - compute perplexity.
    """

    # Testing loop
    batch_loss = 0

    model.eval()
    for data in tqdm(test_loader):
        # get the inputs
        inputs = data.to(device)
        labels = data[:, 1:].to(device)

        # TODO: Run forward pass to get model prediction.
        y_pred = model(inputs)
        y_pred = y_pred[:, :-1, :].permute(0, 2, 1)
        
        # TODO: Compute loss
        loss = criterion(y_pred, labels)
        batch_loss += loss.item()

    test_loss = batch_loss / len(test_loader)
    
    # TODO: Get test perplexity using `test_loss``
    perplexity = math.exp(test_loss)
    print(f'Test loss: {test_loss:.3f}')
    print(f'Test Perplexity: {perplexity:.3f}')
    return test_loss, perplexity

################################################
##       Part3 --- Finetuning          ##
################################################ 

class Encoder(nn.Module):
    def __init__(self, pretrained_encoder, hidden_size):
        super(Encoder, self).__init__()
        self.pretrained_encoder = pretrained_encoder
        self.hidden_size = hidden_size

    def forward(self, input_ids, input_mask):
        # TODO: Implement forward pass.
        # Hint 1: You should take into account the fact that pretrained encoder is bidirectional.
        # Hint 2: Check out the LSTM docs (https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
        # Hint 3: Do we need all the components of the pretrained encoder?
        # TODO CHECK LATER!
        if input_mask is not None:
            input_ids = input_ids.masked_fill(input_mask == 0, 0)
        #print('inside Encoder!')
        #print('input_ids:', input_ids.size())
        embedded_input = self.pretrained_encoder.embedding(input_ids)
        encoder_outputs, (h_n, c_n) = self.pretrained_encoder.lstm(embedded_input)
        #print('encoder_outputs:', encoder_outputs.size(), 'h_n:', h_n.size(), 'c_n:', c_n.size())
        # Since pretrained encoder is bidirectional I concatenated the final hidden state from both directions
        # Extract the last hidden states of the bidirectional LSTM
        # Concatenate the final hidden states from both directions
        encoder_hidden = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        #print('encoder_outputs:', encoder_outputs.size(), 'encoder_hidden:', encoder_hidden.size())
        return encoder_outputs, encoder_hidden

class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size):
        super(AdditiveAttention, self).__init__()
        self.query_weights = nn.Linear(hidden_size, hidden_size)
        self.value_weights = nn.Linear(hidden_size, hidden_size)
        self.combined_weights = nn.Linear(hidden_size, 1)

    def forward(self, query, values, mask):
        # TODO: Implement forward pass.
        # Note: this part requires several lines of code
        #print('AdditiveAttention: query:', query.size(), 'values:', values.size(), 'mask:', mask.size())
        
        query_w = self.query_weights(query).unsqueeze(1)
        values_w = self.value_weights(values)
        #print('AdditiveAttention: query:', query.size(), 'values:', values.size(), 'mask:', mask.size())
        combined_w = self.combined_weights(torch.tanh(query_w + values_w)).squeeze(-1)
        #print('AdditiveAttention: combined_w:', combined_w.size())

        if(mask is not None):
            combined_w = combined_w.masked_fill(mask == 0, float('-inf'))
        
        # Attention weights
        weights = F.softmax(combined_w, dim=1)

        # The context vector is the weighted sum of the values.
        context = torch.sum(weights.unsqueeze(-1) * values, 1)
        #print('AdditiveAttention: context:', context.size(), 'weights:', weights.size())
        return context, weights

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, bos_token_id, dropout_rate=0.15, encoder_embedding=None):
        super(Decoder, self).__init__()
        # Note: feel free to change the architecture of the decoder if you like
        if encoder_embedding is None:
            self.embedding = nn.Embedding(output_size, hidden_size)
        else:
            self.embedding = encoder_embedding
        self.attention = AdditiveAttention(hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(2 * hidden_size, output_size)
        self.bos_token_id = bos_token_id
        self.project_encoder_to_decoder = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, encoder_outputs, encoder_hidden, input_mask,
                target_tensors=None, device=0):
        # TODO: Implement forward pass.
        # Note: this part requires several lines of code
        # Hint: Use target_tensors to handle training and inference appropriately

        if target_tensors is not None: # for training
            input_ids = target_tensors
        else:
            input_ids = torch.tensor([self.bos_token_id] * encoder_outputs.size(0)).unsqueeze(1).to(device)
        #print("Decoder: input_ids:", input_ids.size(), "encoder_hidden:", encoder_hidden.size(), "encoder_outputs:", encoder_outputs.size(), "input_mask:", input_mask.size())

        embedded_input = self.embedding(input_ids)
        embedded_input = self.dropout(embedded_input)
        # I used project_encoder_to_decoder to project the encoder hidden state to the decoder hidden state
        # as the hidden states of the encoder and decoder are of different dimensions (2 * hidden_size and hidden_size respectively)
        decoder_hidden = self.project_encoder_to_decoder(encoder_hidden).unsqueeze(0)
        values = self.project_encoder_to_decoder(encoder_outputs)
        #print("Decoder: embedded_input:", embedded_input.size(), "decoder_hidden_squeeze:", decoder_hidden.size(), "values:", values.size())
        query = decoder_hidden

        # to store all the outputs of the decoder
        decoder_outputs_list = []
        for i in range(embedded_input.size(1)):
            input_i = embedded_input[:, i:i+1, :]
            decoder_output, decoder_hidden = self.gru(input_i, decoder_hidden)
            #print("Decoder: decoder_output:", decoder_output.size(), "decoder_hidden:", decoder_hidden.size(), "query:", query.size(), "values:", values.size(), "input_i:", input_i.size())
            query = decoder_hidden.view(-1, decoder_hidden.size(-1))
            context, _ = self.attention(query, values, input_mask)
            #print("Decoder: decoder_output:", decoder_output.size(), "decoder_hidden:", decoder_hidden.size(), "query:", query.size(), "context:", context.size())
            combined_output = torch.cat((decoder_hidden.permute(1, 0, 2), context.unsqueeze(1)), dim=2)
            #print("Decoder: decoder_output:", decoder_output.size(), "decoder_hidden:", decoder_hidden.size(), "query:", query.size(), "context:", context.size(), "combined_output:", combined_output.size())
            decoder_outputs_list.append(combined_output)

        #context, _ = self.attention(query, values, input_mask)
        #combined_input = torch.cat((embedded_input, context.unsqueeze(1)), dim=2)

        decoder_outputs = torch.cat(decoder_outputs_list, dim=1)
        #print('decoder_outputs:', decoder_outputs.size())

        #decoder_outputs, decoder_hidden = self.gru(combined_input, decoder_hidden.unsqueeze(0))
        decoder_outputs = self.dropout(decoder_outputs)
        decoder_outputs = self.out(decoder_outputs)
        #print("final decoder_outputs:", decoder_outputs.size())
        
        return decoder_outputs, decoder_hidden
    
class EncoderDecoder(nn.Module):
    def __init__(self, hidden_size, input_vocab_size, output_vocab_size, bos_token_id, dropout_rate=0.15, pretrained_encoder=None):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(pretrained_encoder, hidden_size)
        # the embeddings in the encoder and decoder are tied as they're both from the same language
        self.decoder = Decoder(hidden_size, output_vocab_size, bos_token_id, dropout_rate, pretrained_encoder.embedding)

    def forward(self, inputs, input_mask, targets=None):
        encoder_outputs, encoder_hidden = self.encoder(inputs, input_mask)
        decoder_outputs, decoder_hidden = self.decoder(
            encoder_outputs, encoder_hidden, input_mask, targets)
        return decoder_outputs, decoder_hidden

def seq2seq_eval(model, eval_loader, criterion, device=0):
    model.eval()
    epoch_loss = 0
    
    for i, data in tqdm(enumerate(eval_loader), total=len(eval_loader)):
        # TODO: Get the inputs
        input_ids, target_ids, input_mask = data["input_ids"].to(device), data["output_ids"].to(device), data["input_mask"].to(device)

        # TODO: Forward pass
        decoder_outputs, decoder_hidden = model(input_ids, input_mask, target_ids)

        batch_max_seq_length = decoder_outputs.size(1)
        labels = target_ids[:, 1:] # I omited the start token from the labels

        # to padd or truncate the sequence
        decoder_outputs = decoder_outputs[:, :batch_max_seq_length, :]
        labels = labels[:, :batch_max_seq_length]

        # TODO: Compute loss
        loss = criterion(decoder_outputs.view(-1, decoder_outputs.size(-1)), labels.contiguous().view(-1))
        #print("loss:", loss.item())
        epoch_loss += loss.item()

    model.train()

    return epoch_loss / len(eval_loader)

def seq2seq_train(model, train_loader, eval_loader, optimizer, criterion, num_epochs=20, device=0, tensorboard_path="./tensorboard"):
    tb_writer = SummaryWriter(tensorboard_path)
    # Training loop
    model.train()
    best_eval_loss = 1e3 # used to do early stopping

    for epoch in tqdm(range(num_epochs), leave=False, position=0):
        running_loss = 0
        epoch_loss = 0
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader), leave=False, position=1):
            # TODO: Get the inputs
            input_ids, target_ids, input_mask = data["input_ids"].to(device), data["output_ids"].to(device), data["input_mask"].to(device)
            # Forward pass
            decoder_outputs, decoder_hidden = model(input_ids, input_mask, target_ids)
            
            batch_max_seq_length = decoder_outputs.size(1)
            labels = target_ids[:, 1:] # I omited the start token from the labels
            #print("labels:", labels.size())
            #print("decoder_outputs:", decoder_outputs.size())
            #print("decoder_hidden:", decoder_hidden.size())
            #print("input_ids:", input_ids.size())

            # to padd or truncate the sequence
            decoder_outputs = decoder_outputs[:, :batch_max_seq_length - 1, :]
            labels = labels[:, :batch_max_seq_length]
            #print("labels:", labels.size())
            #print("decoder_outputs:", decoder_outputs.size())
            # TODO: Compute loss
            print( "labelsss:", labels.contiguous().view(-1).size())
            print("decoder_outputsss:", decoder_outputs.reshape(-1, decoder_outputs.size(-1)).size())
            print("decoder_outputsss:", decoder_outputs.size())
            print("labels:", labels.contiguous().view(-1)[:130])
            print("decoder_outputsss:", decoder_outputs.reshape(-1, decoder_outputs.size(-1))[0][48470:48480])
            loss = criterion(decoder_outputs.reshape(-1, decoder_outputs.size(-1)), labels.contiguous().view(-1))

            print("loss:", loss.item())
            epoch_loss += loss.item()
            
            # TODO: Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 100 == 99. :    # print every 100 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

        print(f'Epoch {epoch + 1} | Train Loss: {(epoch_loss / len(train_loader)):.4f}')
        eval_loss = seq2seq_eval(model, eval_loader, criterion, device=device)
        print(f'Epoch {epoch + 1} | Eval Loss: {(eval_loss):.4f}')
        tb_writer.add_scalar("ec-finetune/loss/train", epoch_loss / len(train_loader), epoch)
        tb_writer.add_scalar("ec-finetune/loss/eval", eval_loss, epoch)
        
        # TODO: Perform early stopping based on eval loss
        # Make sure to flush the tensorboard writer and close it before returning
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
        else:
            print("Early stopping...")
            break

    tb_writer.flush()
    tb_writer.close()
    return epoch_loss / len(train_loader)

def seq2seq_generate(model, test_loader, tokenizer, device=0):
    generations = []

    for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
        # TODO: get the inputs
        input_ids, target_ids, input_mask = data["input_ids"].to(device), data["output_ids"].to(device), data["input_mask"].to(device)

        # TODO: Forward pass
        outputs, _ = model(input_ids, input_mask, target_ids)

        # TODO: Decode outputs to natural language text
        # Note we expect each output to be a string, not list of tokens here
        for o_id, output in enumerate(outputs):
            generations.append({"input": tokenizer.decode(input_ids[o_id].tolist()),
                                "reference": tokenizer.decode(target_ids[o_id].tolist()), 
                                "prediction": tokenizer.decode(output.argmax(dim=-1).tolist())})
    
    return generations

def evaluate_rouge(generations):
    # TODO: Implement ROUGE evaluation
    references = ...
    predictions = ...

    rouge = evaluate.load('rouge')

    rouge_scores = ...

    return rouge_scores

def t5_generate(dataset, model, tokenizer, device=0):
    # TODO: Implement T5 generation
    generations = []

    for sample in tqdm(dataset, total=len(dataset)):
        reference = ...

        # Hint: use huggingface text generation
        outputs = ...
        prediction = ...
        generations.append({
            "input": ..., 
            "reference": reference, 
            "prediction": prediction})
    
    return generations