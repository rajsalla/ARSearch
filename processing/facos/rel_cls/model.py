# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):   
    def __init__(self, encoder,config,sos_id=None,eos_id=None):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.config=config
        self.sos_id=sos_id
        self.eos_id=eos_id

    def forward(self, input, **kwargs):
        output = self.encoder(input, **kwargs)
        return output


class SimilarityClassifier(nn.Module):
    def __init__(self, encoder, config,sos_id=None,eos_id=None, cuda="cuda:0"):
        super(SimilarityClassifier, self).__init__()
        torch.cuda.set_device(torch.device(cuda))
        self.encoder = Encoder(encoder, config=config, sos_id=sos_id, eos_id=eos_id)

        self.classifier = nn.Sequential(
            nn.Linear(768*2, 32), # fc1
            nn.ReLU(inplace=True), 
            nn.Dropout(p=0.5),
            nn.Linear(32, 8), # fc2
            nn.ReLU(inplace=True), 
            nn.Dropout(p=0.5),
            nn.Linear(8, 2) # fc3
        )

    def forward(self, input1_ids, input1_mask, input2_ids, input2_mask, **inputs_kwargs):
        emb1 = self.encoder(input1_ids, attention_mask=input1_mask)['last_hidden_state'][:,0,:]
        emb2 = self.encoder(input2_ids, attention_mask=input2_mask)['last_hidden_state'][:,0,:]
        aggregated_emb = torch.cat((emb1, emb2), 1)
        output = self.classifier(aggregated_emb)
        return output