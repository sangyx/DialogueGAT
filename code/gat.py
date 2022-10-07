from argparse import ArgumentParser

import math
import dgl
from dgl.nn import GATConv, GINConv, GraphConv, GatedGraphConv
import torch
from torch import nn, optim
from torch.utils import data
from tqdm.auto import tqdm
from torch.nn import functional as F
from torch.cuda.amp import autocast

import pickle
import numpy as np

from sklearn.metrics import mean_squared_error

from data_helper import get_data, collate_fn, ECCDataset


class ContextAttention(nn.Module):
    def __init__(self, in_feats, batch=True):
        super(ContextAttention, self).__init__()
        self.hidden_linear = nn.Linear(in_feats, in_feats)
        self.context_linear = nn.Linear(in_feats, 1, bias=False)
        self.batch = batch

    def forward(self, x):
        h = torch.tanh(self.hidden_linear(x))
        alpha = torch.softmax(self.context_linear(h), dim=0)
        if self.batch:
            alpha = alpha.permute(1, 2, 0)
            attn_x = torch.bmm(alpha, x.transpose(0, 1))[:, 0, :]  # batch, vector
        else:
            attn_x = torch.sum(alpha * x, dim=0)  # batch, vector
        return attn_x, alpha


class DialogueGAT(nn.Module):
    def __init__(
        self,
        vocab,
        W,
        emb_dim,
        n_steps,
        p2gid,
        max_len,
        dropout,
        v_past,
        sen_pos,
        n_heads,
    ):
        super(DialogueGAT, self).__init__()
        self.word_emb = nn.Embedding(len(vocab), emb_dim).from_pretrained(
            torch.tensor(W).float(), freeze=True
        )

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(w, emb_dim))
                for w in [3, 4, 5]
            ]
        )

        self.p2gid = p2gid
        n_parties = len(p2gid)

        self.n_steps = n_steps
        self.party_emb = nn.Embedding(n_parties, emb_dim)
        nn.init.orthogonal_(self.party_emb.weight)

        self.sen_pos = sen_pos
        if self.sen_pos:
            self.sen_pos_encoder = nn.Embedding(max_len, emb_dim)

        self.graph_encoder = nn.ModuleList(
            [GATConv(emb_dim, emb_dim, n_heads, residual=True) for _ in range(n_steps)]
        )
        self.dropout = nn.Dropout(p=dropout)

        self.party_attention = ContextAttention(emb_dim, batch=False)
        self.sen_attention = ContextAttention(emb_dim, batch=False)

        self.v_past = v_past

        if self.v_past:
            self.v_linear = nn.Linear(1, emb_dim)
            self.output = nn.Linear(emb_dim * 3, 1)
        else:
            self.output = nn.Linear(emb_dim * 2, 1)

    def build_graph(self, u_feat, ps):
        u_num = len(u_feat)
        u_edges = [[i, i + 1] for i in range(u_num - 1)] + [
            [i + 1, i] for i in range(u_num - 1)
        ]

        id2q = list(set(ps))
        # print(id2q)
        pid = torch.tensor([self.p2gid[p] for p in id2q], device=u_feat.device)
        q_num = len(id2q)
        q2id = {q: i for i, q in enumerate(id2q)}
        q_edges = []
        for i in range(u_num):
            q_edges.append([i, u_num + q2id[ps[i]]])
            q_edges.append([u_num + q2id[ps[i]], i])

        q_feat = torch.randn(q_num, u_feat.shape[1], device=u_feat.device)
        # q_feat = nn.init.orthogonal_(q_feat.data)

        feat = torch.cat([u_feat, q_feat], dim=0)
        # feat = u_feat
        etypes = torch.tensor([0] * len(u_edges) + [1] * len(q_edges))
        edges = ([e[0] for e in u_edges + q_edges], [e[1] for e in u_edges + q_edges])
        g = dgl.graph(edges)
        g.edata["etypes"] = etypes
        g = g.to(u_feat.device)
        g.ndata["feat"] = feat
        umask = [True] * u_num + [False] * q_num
        lid = torch.arange(0, u_num, 1, device=u_feat.device)
        return g, torch.tensor(umask), u_num, pid, lid

    def batch_graph(self, x, length, ps):
        cl = torch.cumsum(torch.tensor([0] + length), dim=0)
        gs, umask, u_num, pids, lids, ul = [], [], [], [], [], []
        for i in range(0, torch.numel(cl) - 1):
            c_g, c_umask, c_u_num, pid, lid = self.build_graph(
                x[cl[i] : cl[i + 1], :], ps[i]
            )
            gs.append(c_g)
            umask.append(c_umask)
            u_num.append(c_u_num)
            pids.append(pid)
            lids.append(lid)
            ul.append(len(c_umask))

        g = dgl.batch(gs)
        feat = g.ndata["feat"]
        etypes = g.edata["etypes"]
        umask = torch.cat(umask)
        pids = torch.cat(pids)
        lids = torch.cat(lids)
        return g, feat, etypes, umask, pids, lids, ul

    def forward(self, x, length, ps, py):
        emb_x = self.word_emb(x)
        emb_x = emb_x.unsqueeze(1)
        con_x = [conv(emb_x) for conv in self.convs]
        pool_x = [F.max_pool1d(x.squeeze(-1), x.size()[2]) for x in con_x]
        fc_x = torch.cat(pool_x, dim=1)
        wx = fc_x.squeeze(-1)

        g, feat, etypes, umask, pids, lids, ul = self.batch_graph(wx, length, ps)
        feat[~umask, :] = self.party_emb(pids)

        if self.sen_pos:
            feat[umask, :] += self.sen_pos_encoder(lids)

        hs = [feat]
        for i in range(self.n_steps):
            h = self.graph_encoder[i](g, hs[-1])
            h = torch.mean(h, dim=1)
            # h = h.squeeze(dim=1)
            h = self.dropout(h)
            hs.append(h)

        hn = hs[-1]

        ul = torch.cumsum(torch.tensor([0] + ul), dim=0)
        px_list, sx_list = [], []
        for i in range(0, torch.numel(ul) - 1):
            c_hn = hn[ul[i] : ul[i + 1], :]
            c_umask = umask[ul[i] : ul[i + 1]]
            px, pa = self.party_attention(c_hn[~c_umask, :])
            sx, sa = self.sen_attention(c_hn[c_umask, :])
            px_list.append(px)
            sx_list.append(sx)

        px = torch.stack(px_list)
        sx = torch.stack(sx_list)

        if self.v_past:
            ox = torch.cat([px, sx, self.v_linear(py.view(-1, 1))], dim=-1)
        else:
            ox = torch.cat([px, sx], dim=-1)

        out = self.output(ox)
        return out

    def shared_step(self, batch, loss_func, device):
        keys, x, y, ps, length, py = batch
        x, y, py = (x.to(device), y.float().to(device), py.float().to(device))
        y_hat = self(x, length, ps, py)
        loss = loss_func(y_hat, y.reshape(-1, 1))
        return loss, y_hat, y

    def run_epoch(
        self, dataloader, loss_func, optimizer=None, device="cuda", stage="Train",
    ):
        if stage == "Train":
            self.train()
        else:
            self.eval()

        pbar = tqdm(dataloader, desc=stage.ljust(6))
        loss_step, y_prob_step, y_true_step = [], [], []
        for batch in pbar:
            if stage == "Train":
                optimizer.zero_grad()
                loss, y_prob, labels = self.shared_step(batch, loss_func, device)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 15)
                optimizer.step()
            else:
                with torch.no_grad():
                    loss, y_prob, labels = self.shared_step(batch, loss_func, device)

            loss_step.append(loss.item())
            y_prob_step.append(y_prob.data.cpu().numpy())
            y_true_step.append(labels.data.cpu().numpy())

            pbar.set_postfix({"loss": loss_step[-1]})

        pbar.close()
        y_pred = np.concatenate(y_prob_step)
        y_true = np.concatenate(y_true_step)

        metrics = {
            "loss": mean_squared_error(y_true, y_pred),
        }
        return metrics

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--n_steps", default=5, type=int)
        parser.add_argument("--dropout", default=0.1, type=float)
        parser.add_argument("--n_heads", default=5, type=int)
        parser.add_argument("--v_past", action="store_true")
        parser.add_argument("--sen_pos", action="store_true")
        return parser

    @staticmethod
    def build_model(args):
        lr = args.lr
        l2 = args.l2
        dropout = args.dropout
        n_steps = args.n_steps
        sen_pos = args.sen_pos
        n_heads = args.n_heads

        emb_dim = 300

        device = "cuda" if torch.cuda.is_available() and args.use_gpu else "cpu"

        with open("{}data_swd.pkl".format(args.data_path), "rb") as f:
            all_data, vocab, word2idx, W, max_p_len, max_sen_len = pickle.load(f)

        split_data = get_data(all_data, year=args.year)

        gid2p = []
        for s in split_data:
            for key in split_data[s]:
                for t in all_data[key]["transcript"]:
                    gid2p.append(t["name"])
        gid2p = list(set(gid2p))
        p2gid = {p: gid for gid, p in enumerate(gid2p)}

        train_set = ECCDataset(
            split_data["train"], all_data, word2idx, 256, target=args.target,
        )
        val_set = ECCDataset(
            split_data["val"], all_data, word2idx, 256, target=args.target,
        )
        test_set = ECCDataset(
            split_data["test"], all_data, word2idx, 256, target=args.target,
        )

        train_dataloader = data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=4,
        )
        val_dataloader = data.DataLoader(
            val_set,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=4,
        )
        test_dataloader = data.DataLoader(
            test_set,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=4,
        )

        model = DialogueGAT(
            vocab,
            W,
            emb_dim,
            n_steps,
            p2gid,
            252,
            dropout,
            args.v_past,
            sen_pos,
            n_heads,
        )
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

        loss_func = nn.MSELoss()

        return (
            train_dataloader,
            val_dataloader,
            test_dataloader,
            model,
            optimizer,
            loss_func,
            device,
        )
