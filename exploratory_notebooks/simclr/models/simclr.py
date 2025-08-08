import torch.nn as nn
from new_architecture_simclr.network import resnet18, projection_MLP

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, proj_dim=128, hidden_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim)
        )

    def forward(self, x):
        return self.net(x)

# class SimCLRModel(nn.Module):
#     def __init__(self, base_encoder, proj_dim=128):
#         super().__init__()
#         self.encoder = base_encoder
#         self.encoder.fc = nn.Identity()
#         self.projection_head = ProjectionHead(input_dim=CONFIG["FEATURE_DIM"], proj_dim=proj_dim)

#     def forward(self, x):
#         feat = self.encoder(x)
#         proj = self.projection_head(feat)
#         return feat, proj

class SimCLRModel(nn.Module):
    def __init__(self, base_encoder, proj_head):
        super().__init__()
        self.encoder = base_encoder
        self.encoder.fc = nn.Identity()
        self.projection_head = proj_head

    def forward(self, x):
        feat = self.encoder(x)
        proj = self.projection_head(feat)
        return feat, proj
    

def build_simclr_network(DEVICE, args):
    base_encoder = resnet18(
        args,
        num_classes=args.feature_dim,     # make fc output = feature_dim
        zero_init_residual=False
    )
    proj_head = projection_MLP(args)

   
    simclr_model = SimCLRModel(base_encoder, proj_head).to(DEVICE)
    return simclr_model