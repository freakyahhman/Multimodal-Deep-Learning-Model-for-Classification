import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertModel, BertTokenizer
import torch.optim as optim
from tqdm import tqdm

#ImageEncoder
class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.resnet = models.resnet18(preTrained=True) #Call pretrained resnet18
        self.restnet.fc = nn.Identity() #delete the last layer
    def forward(self, x):
        return self.resnet(x)
    
#TextEncoder
class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]
    
#Cross Multi-Head Attention
class Cross_MHA(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Cross_MHA, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    
    def forward(self, query, key, value, key_padding_mask=None):
        attn_output, attn_weights = self.mha(query, key, value, key_padding_mask)
        return attn_output, attn_weights
    
#Add and Norm Layer
class AddNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-6):
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(embed_dim, eps=eps)
    def forward(self, x, sublayer_output):
        return self.norm(x + sublayer_output)

#FusionModule
class FusionModule(nn.Module):
    def __init__(self):
        super(FusionModule, self).__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.text_fc = nn.Linear(768, 512) #map text vector to a vector of 512 dimension that is equal to the image vector
        self.fc_layer = nn.Linear(512, 512) #for the combined vector
        self.final_fc = nn.Linear(512, 16) #very last fc layer to classify
    
    def forward(self, image, input_ids, attention_mask):
        img = self.image_encoder(image) #output of resnet18
        text = self.text_encoder(input_ids, attention_mask) #output of bert
        text = self.text_fc(text) 
        softmax = nn.Softmax(dim=-1) #softmax initialization
        cross_attn = Cross_MHA(512, 8) #Cross Multi-Head Attention with 8 heads
        add_norm = AddNorm() #Layer Normalization
        
        #Text Cross
        query = text
        key = img
        value = img
        text_cross = cross_attn(query, key, value)
        
        #Image Cross
        query = img
        key = text
        value = text
        img_cross = cross_attn(query, key, value)
        
        #Concatenate 2 Vectors
        text_img_combined = torch.cat((text_cross, img_cross), dim=1)
        
        #Self MHA for the Combined Vector
        query = text_img_combined
        key = query
        value = query
        text_img_combined_output = cross_attn(query, key, value)
        
        #Add + Norm
        text_img_combined = add_norm(text_img_combined, text_img_combined_output)
        
        #Feed Forward Layer
        text_img_combined_output = nn.ReLU(self.fc_layer(text_img_combined))
        text_img_combined = add_norm(text_img_combined, text_img_combined_output) #Add and Norm
        
        #Linear Layer that map the vector to a 16 dimension vector, suitable for prediction
        text_img_combined = self.final_fc(text_img_combined)
        output = softmax(text_img_combined) #softmax
        return output
    
#training phase
model = FusionModule()
criterion = nn.CrossEntropyLoss() #Loss function for classification
optimizer = optim.Adam(model.parameters(), lr=0.0001) #Adam Optimization Algorithm
