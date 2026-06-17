from configs.config import config as cfg
from models.decoder import MLP,get_decoder
from models.goten_net import GotenNet,init_parameters,SelfAttentionLayer,HTR,Embedding,cos_cutoff,GATA

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct
from torch_geometric.utils import scatter,softmax
import math

def ace_edge_basis(rbf,r_ij):
    r_ij_ls = torch.cat(r_ij[1:], dim=1)
    return rbf.unsqueeze(-2) * r_ij_ls.unsqueeze(-1)

    A = []
    for r_l in r_ij[1:]:
        A_l = rbf.unsqueeze(-2)*r_l.unsqueeze(-1)
        A.append(A_l)
    A = torch.cat(A, dim=1)
    return A

class DegreeEdgeWeightLayer(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.w_e = nn.Linear(cfg["edge_dim"],cfg["edge_dim"],bias=False)
        self.mlp_e = MLP(in_features=cfg["edge_dim"],out_features=out_features)

    def forward(self, t_ij):
        #t_ij_ls = torch.cat(t_ij, dim=1)
        t_e = self.w_e(t_ij)
        t_e_l2 = torch.norm(t_e, dim=1)
        w = self.mlp_e(t_e_l2)
        return w

class DegreeEdgeAttentionLayer(nn.Module):
    def __init__(self, out_features,hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = out_features

        self.w_eq = nn.ModuleList([nn.Linear(cfg["edge_dim"], hidden_dim, bias=False) for _ in range(cfg["degree_max"])])
        self.w_ek = nn.ModuleList([nn.Linear(cfg["edge_dim"], hidden_dim, bias=False) for _ in range(cfg["degree_max"])])
        self.mlp_e = MLP(in_features=hidden_dim,out_features=out_features)

    def forward(self, t_ij):
        w_eqs = torch.stack([w_l.weight for w_l in self.w_eq])[cfg["high_degree_sizes"][1]]
        eq = torch.einsum('ecf, ckf -> eck', t_ij, w_eqs)
        #eq = self.w_eq(t_ij)
        w_eks = torch.stack([w_l.weight for w_l in self.w_ek])[cfg["high_degree_sizes"][1]]
        ek = torch.einsum('ecf, ckf -> eck', t_ij, w_eks)
        w = self.mlp_e((eq * ek).sum(dim=1))
        return w

class MyNet(GotenNet):
    def __init__(self, out_label=None, mean=0, std=1, net_type=None):
        super().__init__(out_label,mean,std,"exp",False)

        self.gata_list = nn.ModuleList()
        match net_type:
            case "mace":
                self.embedding = MaceEdgeEmbedding(rbf_type="exp")
                for i in range(cfg["layer_num"]):
                    self.gata_list.append(MySEAOnlyMultibodyGATA(init_X=i == 0, is_last=i == cfg["layer_num"] - 1,scalar_mult=False))

                for gata in self.gata_list:
                    gata.sea = EdgeSEA(gata.S * cfg["node_dim"], cfg["node_dim"])
            case "mace_scalar":
                self.embedding = MaceEdgeEmbedding(rbf_type="exp")
                for i in range(cfg["layer_num"]):
                    self.gata_list.append(MySEAOnlyMultibodyGATA(init_X=i == 0, is_last=i == cfg["layer_num"] - 1, scalar_mult=True))

                for gata in self.gata_list:
                    gata.sea = EdgeSEA(gata.S * cfg["node_dim"], cfg["node_dim"])
            case "ace_edge":
                self.embedding = MaceEdgeEmbedding(rbf_type="exp")
                for i in range(cfg["layer_num"]):
                    self.gata_list.append(MySEAOnlyGATA(init_X=i == 0, is_last=i == cfg["layer_num"] - 1))

                for gata in self.gata_list:
                    gata.sea = EdgeSEA(gata.S * cfg["node_dim"], cfg["node_dim"])
            case "ace":
                self.embedding = MaceBasisEmbedding(rbf_type="exp")
                for i in range(cfg["layer_num"]):
                    self.gata_list.append(MySEAOnlyGATA(init_X=i == 0, is_last=i == cfg["layer_num"] - 1))

                for gata in self.gata_list:
                    gata.sea = MaceSEA(gata.S * cfg["node_dim"], cfg["node_dim"])
                    gata.htr = MaceHTR() if gata.htr is not None else None
            case "ace_no_htr":
                self.embedding = MaceBasisEmbedding(rbf_type="exp")
                for i in range(cfg["layer_num"]):
                    self.gata_list.append(MySEAOnlyGATA(init_X=i == 0, is_last=True))

                for gata in self.gata_list:
                    gata.sea = MaceSEA(gata.S * cfg["node_dim"], cfg["node_dim"])
            case other:
                raise NotImplementedError("Unknown modified net type: {}".format(other))

        #for gata in self.gata_list:
        #    gata.sea = MySEA(gata.S * cfg["node_dim"], cfg["node_dim"])
        #    gata.htr = MyHTR() if gata.htr is not None else None

        self.apply(init_parameters)
        self.set_decoder(self.out_label, mean, std)

    def set_decoder(self, decoder_type, mean=0, std=1, need_guide=False):
        if decoder_type in ["alpha","zpve","u0","u","h","g","cv"]:
            decoder_type = "scalar_mix"
        elif decoder_type in ["homo","lumo"]:
            decoder_type = "scalar_int"
        self.decoder = get_decoder(decoder_type, mean, std, need_guide)
        self.decoder_type = decoder_type

class MaceEdgeEmbedding(Embedding):
    def __init__(self, rbf_type="exp"):
        super().__init__(rbf_type)
        self.w_ace = DegreeEdgeWeightLayer(cfg["edge_dim"])
        self.w_emp = nn.Linear(cfg["rbf_num"], cfg["edge_dim"])

    def forward(self, z, p, edge_index):
        '''
        :param z: atomic number
        :param p: relative position of nodes
        :return:
        '''

        # Edge Tensor Representation
        r_0 = torch.norm(p, dim=1).unsqueeze(1)
        r_ij = [r_0] + list(torch.split(self.sphere(p / r_0)[:, 1:], cfg["high_degree_sizes"][0], dim=1))

        n_j, n_i = edge_index
        rbf_0 = self.rbf(r_0)

        # Node Scalar Feature Initialization
        m_i = scatter(self.a_nbr(z).squeeze(1)[n_j] * (self.w_ndp(rbf_0) * cos_cutoff(r_0)), n_i, dim=0, dim_size=len(z), reduce="sum")
        h = self.w_nrd_nru(torch.cat([self.a_na(z).squeeze(1), m_i], dim=1))

        # Edge Scalar Feature Initialization
        ace_basis = ace_edge_basis(self.w_emp(rbf_0),r_ij)
        t_ij = (h[n_i] + h[n_j]) * self.w_erp(rbf_0) * self.w_ace(ace_basis)

        # High-degree Steerable Feature will be initialized in the first GATA module
        X = None

        return r_ij, h, t_ij, X

class MaceBasisEmbedding(Embedding):
    def __init__(self, rbf_type="exp"):
        super().__init__(rbf_type)

    def forward(self, z, p, edge_index):
        '''
        :param z: atomic number
        :param p: relative position of nodes
        :return:
        '''

        #Edge Tensor Representation
        r_0 = torch.norm(p, dim=1).unsqueeze(1)
        r_ij = [r_0] + list(torch.split(self.sphere(p/r_0)[:, 1:], cfg["high_degree_sizes"][0], dim=1))

        n_j,n_i = edge_index
        rbf_0 = self.rbf(r_0)

        # Node Scalar Feature Initialization
        m_i = scatter(self.a_nbr(z).squeeze(1)[n_j] * (self.w_ndp(rbf_0) * cos_cutoff(r_0)), n_i, dim=0, dim_size=len(z),reduce="sum")
        h = self.w_nrd_nru(torch.cat([self.a_na(z).squeeze(1),m_i],dim=1))

        # Edge Scalar Feature Initialization
        t_ij = ace_edge_basis(self.w_erp(rbf_0),r_ij)

        # High-degree Steerable Feature will be initialized in the first GATA module
        X = None

        return r_ij,h,t_ij,X

class MyEmbedding(Embedding):
    def __init__(self, rbf_type="exp"):
        super().__init__(rbf_type)
        #self.mlp_hs = MLP(in_features=cfg["node_dim"],out_features=cfg["node_dim"])
        self.w_edp = nn.Linear(cfg["rbf_num"],cfg["edge_dim"])

    def forward(self, z, p, edge_index):
        '''
        :param z: atomic number
        :param p: relative position of nodes
        :return:
        '''

        #Edge Tensor Representation
        r_0 = torch.norm(p, dim=1).unsqueeze(1)
        r_ij = [r_0] + list(torch.split(self.sphere(p/r_0)[:, 1:], cfg["high_degree_sizes"][0], dim=1))

        n_j,n_i = edge_index
        rbf_0 = self.rbf(r_0)

        # Node Scalar Feature Initialization
        m_i = scatter(self.a_nbr(z).squeeze(1)[n_j] * (self.w_ndp(rbf_0) * cos_cutoff(r_0)), n_i, dim=0, dim_size=len(z),reduce="sum")
        h = self.w_nrd_nru(torch.cat([self.a_na(z).squeeze(1),m_i],dim=1))

        # Edge Scalar Feature Initialization
        t_ij = ace_edge_basis(self.w_edp(rbf_0),r_ij)
        #t_ij = (h[n_i]+h[n_j]) * self.w_erp(rbf_0)

        # High-degree Steerable Feature will be initialized in the first GATA module
        X = None

        return r_ij,h,t_ij,X

class EdgeSEA(SelfAttentionLayer):
    def __init__(self, out_features, attn_dim=None):
        attn_dim = out_features if attn_dim is None else attn_dim
        super().__init__(attn_dim)
        assert attn_dim % self.head_num == 0
        self.out_features = out_features

        self.w_q = nn.Linear(cfg["node_dim"], attn_dim)
        self.w_k = nn.Linear(cfg["node_dim"], attn_dim)

        self.mlp_v = MLP(in_features=cfg["node_dim"], out_features=attn_dim)

        self.w_re = nn.Linear(cfg["edge_dim"], attn_dim)
        self.w_rs = nn.Linear(cfg["edge_dim"], attn_dim)

        self.combine_heads = nn.Linear(in_features=attn_dim, out_features=out_features, bias=True)

    def forward(self, h, t_ij, r_ij, edge_index):
        n_j, n_i = edge_index

        q_i = self.w_q(h).reshape(h.shape[0],self.head_num,-1)[n_i]
        k_j = self.w_k(h).reshape(h.shape[0],self.head_num,-1)[n_j]
        v_j = self.mlp_v(h).reshape(h.shape[0],self.head_num,-1)[n_j]
        w_j = self.act_fn(self.w_re(t_ij)).reshape(k_j.shape[0],self.head_num,-1)

        a_ij = (q_i * (k_j * w_j)).sum(dim=-1,keepdims=True)
        a_ij = softmax(a_ij, n_i, num_nodes=h.shape[0], dim=0)

        # not in the paper but in the author's code
        #norm = 1.0 / math.sqrt(cfg["node_dim"])
        n_i_edges = scatter(torch.ones([len(n_i),1,1],device=n_i.device),n_i,dim=0,reduce="sum")
        norm = torch.sqrt(n_i_edges) / math.sqrt(cfg["node_dim"])
        norm = norm[n_i]
        a_ij = a_ij * norm

        a_ij = self.dropout(a_ij)
        sea_ij = a_ij * v_j #[E,H,D*5/H]
        sea_ij = sea_ij.flatten(1) #[E,D*5]
        sea_ij = sea_ij + v_j.flatten(1) * self.w_rs(t_ij) * cos_cutoff(r_ij[0])

        sea_ij = self.combine_heads(sea_ij)
        return sea_ij

class MaceSEA(EdgeSEA):
    def __init__(self, out_features, attn_dim=None):
        attn_dim = out_features if attn_dim is None else attn_dim
        super().__init__(out_features, attn_dim)

        self.w_re = DegreeEdgeWeightLayer(attn_dim)
        self.w_rs = DegreeEdgeWeightLayer(attn_dim)

class MySEA(SelfAttentionLayer):
    def __init__(self, out_features, attn_dim=None):
        attn_dim = out_features if attn_dim is None else attn_dim
        super().__init__(attn_dim)
        assert attn_dim % self.head_num == 0
        self.out_features = out_features

        self.w_q = nn.Linear(cfg["node_dim"], attn_dim)
        self.w_k = nn.Linear(cfg["node_dim"], attn_dim)

        self.mlp_v = MLP(in_features=cfg["node_dim"], out_features=attn_dim)

        self.w_re = DegreeEdgeWeightLayer(attn_dim)
        self.w_rs = DegreeEdgeWeightLayer(attn_dim)

        self.combine_heads = nn.Linear(in_features=attn_dim, out_features=out_features, bias=True)

    def forward(self, h, t_ij, r_ij, edge_index):
        n_j, n_i = edge_index

        q_i = self.w_q(h).reshape(h.shape[0],self.head_num,-1)[n_i]
        k_j = self.w_k(h).reshape(h.shape[0],self.head_num,-1)[n_j]
        v_j = self.mlp_v(h).reshape(h.shape[0],self.head_num,-1)[n_j]
        w_j = self.act_fn(self.w_re(t_ij)).reshape(k_j.shape[0],self.head_num,-1)

        a_ij = (q_i * (k_j * w_j)).sum(dim=-1,keepdims=True)
        a_ij = softmax(a_ij, n_i, num_nodes=h.shape[0], dim=0)

        # not in the paper but in the author's code
        #norm = 1.0 / math.sqrt(cfg["node_dim"])
        n_i_edges = scatter(torch.ones([len(n_i),1,1],device=n_i.device),n_i,dim=0,reduce="sum")
        norm = torch.sqrt(n_i_edges) / math.sqrt(cfg["node_dim"])
        norm = norm[n_i]
        a_ij = a_ij * norm

        a_ij = self.dropout(a_ij)
        sea_ij = a_ij * v_j #[E,H,D*5/H]
        sea_ij = sea_ij.flatten(1) #[E,D*5]
        sea_ij = sea_ij + v_j.flatten(1) * self.w_rs(t_ij) * cos_cutoff(r_ij[0])

        sea_ij = self.combine_heads(sea_ij)
        return sea_ij

class MaceHTR(HTR):
    def __init__(self):
        super().__init__()
        #self.head_num = cfg["attention_heads"]

        #self.mlp_w = None
        self.mlp_t = None
        self.w_t = nn.Linear(in_features=cfg["edge_dim"],out_features=cfg["edge_dim"],bias=False)
        #self.mlp_h = nn.Linear(in_features=cfg["node_dim"], out_features=cfg["edge_ref_dim"])

    def forward(self, h, X, t_ij, r_ij, edge_index, batch_index):
        n_j, n_i = edge_index
        X_ls = torch.cat(X, dim=1)  # [N,8,256]

        eq_i = self.w_vq(X_ls)[n_i]  # [E,8,256]
        w_vks = torch.stack([w_l.weight for w_l in self.w_vk])[cfg["high_degree_sizes"][1]]
        ek_j = torch.einsum('ecf, ckf -> eck', X_ls, w_vks)[n_j]  # [E,8,256]

        w_ij = eq_i * ek_j  # [E,256]

        dt_ij = self.mlp_w(w_ij) * self.w_t(t_ij)  # [E,256]
        return dt_ij

class MySEAOnlyGATA(GATA):
    def __init__(self,init_X=False,is_last=False):
        super().__init__(init_X,is_last)
        self.w_rs = None
        self.mlp_s = None

    def forward(self, h, X, t_ij, r_ij, edge_index, batch_index):
        n_j, n_i = edge_index
        r_0 = r_ij[0]

        sea_ij = self.sea(h, t_ij, r_ij, edge_index)

        o_ij = sea_ij
        o_ij = o_ij.view(edge_index.shape[1], -1, cfg["node_dim"])

        o_ij_s = o_ij[:, 0, :]
        dh = scatter(o_ij_s, n_i, dim=0, dim_size=len(h), reduce="sum")
        h = h + dh

        o_ij_d = o_ij[:, 1:1 + cfg["degree_max"], :]
        o_ij_t = None
        if X is not None:
            o_ij_t = o_ij[:, 1 + cfg["degree_max"]:1 + cfg["degree_max"] * 2, :]
        else:
            X = []

        for i in range(cfg["degree_max"]):
            l = i + 1
            if o_ij_t is not None:
                dX_l = scatter(o_ij_d[:, i:i + 1, :] * r_ij[l].unsqueeze(-1) + o_ij_t[:, i:i + 1, :] * X[i][n_j], n_i, dim=0, dim_size=len(h), reduce="sum")
                X[i] = X[i] + dX_l
            else:
                dX_l = scatter(o_ij_d[:, i:i + 1, :] * r_ij[l].unsqueeze(-1), n_i, dim=0, dim_size=len(h), reduce="sum")
                X.append(dX_l)

        if self.htr is not None:
            dt_ij = self.htr(h, X, t_ij, r_ij[1:], edge_index, batch_index)
            t_ij = t_ij + dt_ij

        return h, X, t_ij

class MySEAOnlyMultibodyGATA(GATA):
    def __init__(self,init_X=False,is_last=False,scalar_mult=False):
        super().__init__(init_X,is_last)
        self.w_rs = None
        self.mlp_s = None

        self.body_max = 4
        self.w_hb = nn.ModuleList([nn.Linear(cfg["node_dim"], cfg["node_dim"]) for _ in range(2,self.body_max)]) if scalar_mult else []
        c_n = sum(cfg["high_degree_sizes"][0])
        self.w_Xb = nn.ModuleList([nn.Linear(c_n*cfg["node_dim"], c_n*cfg["node_dim"],bias=False) for _ in range(2, self.body_max)])
        self.irreps = o3.Irreps("+".join(["{}x{}o".format(cfg["node_dim"], l+1) for l in range(cfg["degree_max"])]))
        self.tp = [FullyConnectedTensorProduct(
            self.irreps,
            self.irreps,
            self.irreps
        ) for _ in self.w_Xb]

    def forward(self, h, X, t_ij, r_ij, edge_index, batch_index):
        n_j, n_i = edge_index
        r_0 = r_ij[0]

        sea_ij = self.sea(h, t_ij, r_ij, edge_index)

        o_ij = sea_ij
        o_ij = o_ij.view(edge_index.shape[1], -1, cfg["node_dim"])

        o_ij_s = o_ij[:, 0, :]
        dh_2b = scatter(o_ij_s, n_i, dim=0, dim_size=len(h), reduce="sum")
        dh = dh_2b
        for i,w_h_b in enumerate(self.w_hb):
            b_i = i+3
            dh = dh + w_h_b((dh_2b**(b_i-1))/math.sqrt(math.factorial(b_i-1)))
        h = h + dh

        o_ij_d = o_ij[:, 1:1 + cfg["degree_max"], :]
        o_ij_t = None
        if X is not None:
            o_ij_t = o_ij[:, 1 + cfg["degree_max"]:1 + cfg["degree_max"] * 2, :]
        else:
            X = []

        dX = []
        for i in range(cfg["degree_max"]):
            l = i + 1
            if o_ij_t is not None:
                dX_l = scatter(o_ij_d[:, i:i + 1, :] * r_ij[l].unsqueeze(-1) + o_ij_t[:, i:i + 1, :] * X[i][n_j], n_i, dim=0, dim_size=len(h), reduce="sum")
            else:
                dX_l = scatter(o_ij_d[:, i:i + 1, :] * r_ij[l].unsqueeze(-1), n_i, dim=0, dim_size=len(h), reduce="sum")
            dX.append(dX_l)

        # 2-body => n-body
        dX_ls = torch.cat(dX, dim=1)
        dX_b = dX_2b = dX_ls.reshape(h.shape[0],-1) #torch.cat([dX_l.sum(dim=1) for dX_l in dX], dim=1)
        for i,w_X_b in enumerate(self.w_Xb):
            dX_b = self.tp[i](dX_b, dX_2b)
            dX_ls = dX_ls + w_X_b(dX_b).reshape(h.shape[0], -1, cfg["node_dim"])
        dX = list(torch.split(dX_ls, cfg["high_degree_sizes"][0], dim=1))

        for i,dX_l in enumerate(dX):
            if o_ij_t is not None:
                X[i] = X[i] + dX_l
            else:
                X.append(dX_l)

        if self.htr is not None:
            dt_ij = self.htr(h, X, t_ij, r_ij[1:], edge_index, batch_index)
            t_ij = t_ij + dt_ij

        return h, X, t_ij