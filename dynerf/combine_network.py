import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from activation import trunc_exp
from .renderer import NeRFRenderer
from .network import DyNeRFNetwork
from torch.autograd import Variable

class StaticNerf(nn.Module):
    def __init__(self,
                 encoding="frequency",
                 encoding_dir="frequency",
                 num_layers=6,
                 hidden_dim=256,
                 num_layers_color=3,
                 bound=1
                ):
        super().__init__()

        # sigma network
        self.bound = bound
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.encoder, self.in_dim = get_encoder(encoding, desired_resolution=2048 * bound, multires = 12)
        # time encodeing L should be more complicated

        sigma_net = []
        for l in range(num_layers):
            if l == 0 :
                in_dim = self.in_dim
            elif l == num_layers // 2:
                in_dim = self.hidden_dim + self.in_dim                
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1 + self.hidden_dim # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim
            
            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = nn.ModuleList(sigma_net)

        # color network
        self.num_layers_color = num_layers_color        
        self.encoder_dir, self.in_dim_color = get_encoder(encoding_dir, multires = 8)
        self.in_dim_color += (self.hidden_dim)
        
        color_net =  []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.in_dim_color
            else:
                in_dim = hidden_dim
            
            if l == num_layers_color - 1:
                out_dim = 3 # 3 rgb
            else:
                out_dim = hidden_dim
            
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.color_net = nn.ModuleList(color_net)

    
    def forward(self, x, d, t):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        # t: [N,1], normallized in [0, 1]

        # sigma
        x = self.encoder(x, bound=self.bound)

        h = x
        for l in range(self.num_layers):
            if l == self.num_layers // 2:
                h = torch.cat([h,x], dim = -1) 
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        sigma = h[..., 0]
        geo_feat = h[..., 1:]

        # color
        
        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        color = h

        return sigma, color

    def density(self, x, t):
        # x: [N, 3], in [-bound, bound]

         # sigma
        x = self.encoder(x, bound=self.bound)

        h = x
        for l in range(self.num_layers):
            if l == self.num_layers // 2:
                h = torch.cat([h,x], dim = -1) 
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        sigma = h[..., 0]
        geo_feat = h[..., 1:]

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
        }

    # allow masked inference
    def color(self, x, d, t, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.

        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device) # [N, 3]
            # in case of empty mask
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
            t = t[mask]
            geo_feat = geo_feat[mask]

        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype) # fp16 --> fp32
        else:
            rgbs = h

        return rgbs


class DynamicNerf(nn.Module):
    def __init__(self,
                 encoding="frequency",
                 encoding_dir="frequency",
                 num_layers=6,
                 hidden_dim=256,
                 num_layers_color=3,
                 bound=1
                 ):
        super().__init__()

        # sigma network
        self.bound = bound
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.encoder, self.in_dim = get_encoder(encoding, desired_resolution=2048 * bound, multires = 12)
        # time encodeing L should be more complicated
        self.encoder_time, self.in_dim_time = get_encoder("frequency", input_dim = 1, multires = 12)

        sigma_net = []
        for l in range(num_layers):
            if l == 0 :
                in_dim = self.in_dim + self.in_dim_time
            elif l == num_layers // 2:
                in_dim = self.hidden_dim + self.in_dim_time + self.in_dim                
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1 + self.hidden_dim # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim
            
            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = nn.ModuleList(sigma_net)

        # color network
        self.num_layers_color = num_layers_color        
        self.encoder_dir, self.in_dim_color = get_encoder(encoding_dir, multires = 8)
        self.in_dim_color += (self.hidden_dim)
        
        color_net =  []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.in_dim_color
            else:
                in_dim = hidden_dim
            
            if l == num_layers_color - 1:
                out_dim = 3 # 3 rgb
            else:
                out_dim = hidden_dim
            
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.color_net = nn.ModuleList(color_net)
    
    def forward(self, x, d, t):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        # t: [N,1], normallized in [0, 1]

        # sigma
        x = self.encoder(x, bound=self.bound)
        t = self.encoder_time(t)

        h = x
        for l in range(self.num_layers):
            if l == 0:
                h = torch.cat([h,t], dim = -1)
            elif l == self.num_layers // 2:
                h = torch.cat([h,x,t], dim = -1) 
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        sigma = h[..., 0]
        geo_feat = h[..., 1:]

        # color
        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        color = h

        return sigma, color

    def density(self, x, t):
        # x: [N, 3], in [-bound, bound]

         # sigma
        x = self.encoder(x, bound=self.bound)
        t = self.encoder_time(t)

        h = x
        for l in range(self.num_layers):
            if l == 0:
                h = torch.cat([h,t], dim = -1)
            elif l == self.num_layers // 2:
                h = torch.cat([h,x,t], dim = -1) 
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        sigma = h[..., 0]
        geo_feat = h[..., 1:]

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
        }

    # allow masked inference
    def color(self, x, d, t, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.

        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device) # [N, 3]
            # in case of empty mask
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
            t = t[mask]
            geo_feat = geo_feat[mask]

        d = self.encoder_dir(d)
        t = self.encoder_time(t)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype) # fp16 --> fp32
        else:
            rgbs = h

        return rgbs

class CombineDyNeRFNetwork(NeRFRenderer):
    def __init__(self,
                 encoding="frequency",
                 encoding_dir="frequency",
                 num_layers=6,
                 hidden_dim=256,
                 num_layers_color=3,
                 bound=1,
                 static = False,
                 **kwargs,
                 ):
        super().__init__(bound, **kwargs)

        self.static = static
        self.static_model = StaticNerf(bound = bound)
        self.dynamic_model = DynamicNerf(bound = bound)
    
    def forward(self, x, d, t):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        # t: [N,1], normallized in [0, 1]

        # sigma
        if self.static:
            s_sigma, s_color = self.static_model(x, d, t)
            return trunc_exp(s_sigma), torch.sigmoid(s_color)
        else:
            with torch.no_grad():
                s_sigma, s_color = self.static_model(x, d, t)
            d_sigma, d_color = self.dynamic_model(x, d, t)
            return trunc_exp(s_sigma + d_sigma), torch.sigmoid(s_color + d_color)

    def density(self, x, t):
        
        if self.static:
            s_info = self.static_model.density(x, t)
            return {
                "sigma" : trunc_exp(s_info["sigma"]),
                "geo_feat" : s_info["geo_feat"],
                "s_geo_feat" : s_info["geo_feat"],
                "d_geo_feat" : s_info["geo_feat"]
            }

        else:
            with torch.no_grad():
                s_info = self.static_model.density(x, t)
            d_info = self.dynamic_model.density(x, t)
            return {
                "sigma" : trunc_exp(s_info["sigma"] + d_info["sigma"]),
                "geo_feat" : s_info["geo_feat"] + d_info["geo_feat"],
                "s_geo_feat" : s_info["geo_feat"],
                "d_geo_feat" : d_info["geo_feat"]
            }
            

    # allow masked inference
    def color(self, x, d, t, mask=None, geo_feat=None, s_geo_feat=None, d_geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.

        if self.static:
            s_rgbs = self.static_model.color(x, d, t, mask = mask, geo_feat = s_geo_feat)
            return torch.sigmoid(s_rgbs)
            
        else:
            with torch.no_grad():
                s_rgbs = self.static_model.color(x, d, t, mask = mask, geo_feat = s_geo_feat)
            d_rgbs = self.dynamic_model.color(x, d, t, mask = mask, geo_feat = d_geo_feat)
            return torch.sigmoid(s_rgbs + d_rgbs)