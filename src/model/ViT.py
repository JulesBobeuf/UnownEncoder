import torch
from model.VisionEncoder import VisionEncoder
import utils.constants as constants

class ViT(torch.nn.Module):
    def __init__(self, image_size, channel_size, patch_size, embed_size, nb_heads, classes, nb_layers, hidden_size, dropout):
        super(ViT, self).__init__()

        self.patch_size = patch_size
        self.embed_size = embed_size
        self.nb_patches = (image_size // patch_size) ** 2
        self.pixels_per_patch = channel_size * (patch_size ** 2)
        self.nb_heads = nb_heads
        self.classes = classes
        self.nb_layers = nb_layers
        self.hidden_size = hidden_size
        self.dropout = dropout

        # Layers
        self.embedding = torch.nn.Linear(self.pixels_per_patch,embed_size)
        
        self.positional_encoding = torch.nn.Parameter(torch.randn(1, self.nb_patches+1, self.embed_size))
        self.dropout_layer = torch.nn.Dropout(dropout)
        self.class_token = torch.nn.Parameter(torch.randn(1, 1, self.embed_size))

        self.encoders = torch.nn.ModuleList([])        
        for i in range(self.nb_layers):
            self.encoders.append(VisionEncoder(self.embed_size, self.nb_heads, self.hidden_size, self.dropout))

        self.norm = torch.nn.LayerNorm(embed_size)
        self.classifier = torch.nn.Linear(self.embed_size, self.classes)
        
        
    def forward(self, img_torch):
        b, c, h, w = img_torch.size()
        img_torch_reshape = x = img_torch.reshape(b, int((h / self.patch_size) * (w / self.patch_size)), c * self.patch_size * self.patch_size).float()
        fwd_embeddings = self.embedding(img_torch_reshape)
        
        b,n,e = fwd_embeddings.size()
        class_token = torch.randn(b,1,e).to(constants.DEVICE)
        fwd_cat_class_token = torch.cat((fwd_embeddings, class_token),1).to(constants.DEVICE)
        
        fwd_pos_encoding = (fwd_cat_class_token+ self.positional_encoding).to(constants.DEVICE)
        fwd_dropout = self.dropout_layer(fwd_pos_encoding).to(constants.DEVICE)
        x = fwd_dropout
        for encoder in self.encoders:
            x = encoder(x)
        
        fwd_encoders = x
        fwd_cls = fwd_encoders[:, -1, :]
        fwd_norm = self.norm(fwd_cls)
        fwd_classifier = self.classifier(fwd_norm)
        fwd_softmax = torch.nn.functional.log_softmax(fwd_classifier, dim=1)
        
        return fwd_softmax