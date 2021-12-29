import torch, torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc


class Model_fn():
    def __init__(self, args):
        self.device = args.device
        self.loss_fun = torch.nn.BCELoss()
        self.train_num = 0

    def fit(self, train_loader, optimizer):
        train_loader.dataset.reset()
        self.train()
        self.to(self.device)
        description = "Training (the {:d}-batch): tra_Loss = {:.4f}"
        loss_total, avg_loss = 0.0, 0.0
        epochs = tqdm(train_loader, leave=False, desc='local_update')
        for idx, batch in enumerate(epochs):
            optimizer.zero_grad()
            batch = to_device(batch, self.device)
            output = self(batch)
            label = batch['label'].float()
            loss = self.loss_fun(output.squeeze(-1), label)
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
            avg_loss = loss_total / (idx + 1)
            epochs.set_description(description.format(idx + 1, avg_loss))
        self.train_num = len(train_loader.dataset)

    def evaluate(self, test_loader, recorder=None):
        test_loader.dataset.reset()
        self.eval()
        self.to(self.device)
        loss_total = 0.0
        label, pred = [], []
        with torch.no_grad():
            with tqdm(test_loader) as epochs:
                for idx, batch in enumerate(epochs):
                    batch = to_device(batch, self.device)
                    output = self(batch)
                    pred += output.squeeze(-1).tolist()
                    label += batch['label'].tolist()
                    loss = self.loss_fun(
                        output.squeeze(-1), batch['label'].float())
                    loss_total += loss.item()
            loss_avg = loss_total/len(test_loader)
            fpr, tpr, _ = roc_curve(label, pred)
            auc_score = auc(fpr, tpr)
        recorder['loss'].append(loss_avg)
        recorder['auc'].append(auc_score)


class DIN(nn.Module, Model_fn):
    def __init__(self, args):
        super(DIN, self).__init__()

        self.args = args
        self.features = args.features
        self._estimator_type = 'classifier'
        self.num_inputs = nn.ModuleDict()
        self.embeddings = nn.ModuleDict()
        self.cat_embeddings = nn.ModuleDict()
        self.seq_embeddings = nn.ModuleDict()
        cat_size = 0

        for embed_key in args.embedding.keys():
            self.embeddings[embed_key] = nn.Embedding(
                args.embedding[embed_key]['num'], args.embedding[embed_key]['size'])
            for feats_key, feats_value in args.use_feats.items():
                if embed_key in feats_key:
                    if feats_value == 'cat_feats':
                        self.cat_embeddings[feats_key] = self.embeddings[embed_key]
                    if feats_value == 'seq_feats':
                        self.seq_embeddings[feats_key] = self.embeddings[embed_key]
                    cat_size += args.embedding[embed_key]['size']
        args.item_embed_size = sum(
            [v['size'] for k, v in args.embedding.items() if 'item' in k])

        for key in self.features['num_feats']:
            self.num_inputs[key] = nn.Identity()
            cat_size += 1
    
        self.pooling = Pooling('attention', dim=1, args=args)
        self.mlp = MLP(cat_size, self.args)

        Model_fn.__init__(self, args)

    def forward(self, inputs):

        embedded = {}
        for key, module in self.num_inputs.items():
            out = module(inputs[key]).unsqueeze(-1)
            embedded[key] = out
        can_embedded, exp_embedded, ipv_embedded = [], [], []
        for key, module in self.cat_embeddings.items():
            out = module(inputs[key])
            if 'cand_item' in key:
                can_embedded.append(out)
            else:
                embedded[key] = out
        embedded['cand_item'] = torch.cat(can_embedded, dim=1)
        for key, module in self.seq_embeddings.items():
            seq_out = module(inputs[key])
            if 'exp_item' in key:
                exp_embedded.append(seq_out)
            elif 'ipv_item' in key:
                ipv_embedded.append(seq_out)

        exp_seq = torch.cat(exp_embedded, dim=-1)
        exp_out = self.pooling(exp_seq, embedded['cand_item'])
        embedded['exp_item'] = exp_out

        ipv_seq = torch.cat(ipv_embedded, dim=-1)
        ipv_out = self.pooling(ipv_seq, embedded['cand_item'])
        embedded['ipv_item'] = ipv_out

        emb_cat = torch.cat(list(embedded.values()), dim=1)
        score_logits = -torch.log(1/inputs['score'].unsqueeze(-1)-1)
        output = torch.sigmoid(self.mlp(emb_cat)+score_logits)
        return output


class MLP(nn.Module):
    def __init__(self, input_size, args):
        super(MLP, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_size, args.hidden_size[0]),
            nn.BatchNorm1d(args.hidden_size[0])
        )
        self.fc2 = nn.Sequential(
            nn.Linear(args.hidden_size[0], args.hidden_size[1]),
            nn.BatchNorm1d(args.hidden_size[1])
        )
        self.fc3 = nn.Linear(args.hidden_size[1], 1)
        self.relu = torch.nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, input):
        x = self.dropout(self.relu(self.fc1(input)))
        x = self.dropout(self.relu(self.fc2(x)))
        output = self.fc3(x)
        return output


class Pooling(nn.Module):
    def __init__(self, pooling_type, dim=1, **kwargs):
        super(Pooling, self).__init__()
        self.dim = dim
        self.pooling_type = pooling_type
        if self.pooling_type == 'mean':
            self.pooling = torch.mean
        if self.pooling_type == 'sum':
            self.pooling = torch.sum
        if self.pooling_type == 'attention':
            self.pooling = Attention_Pooling(kwargs['args'])

    def forward(self, x, target_item=None):
        if self.pooling_type != 'attention':
            output = self.pooling(x, self.dim)
        else:
            output = self.pooling(x, target_item, self.dim)
        return output


class Attention_Pooling(nn.Module):
    def __init__(self, args):
        super(Attention_Pooling, self).__init__()
        self.attention_unit = Attention_Unit(args)

    def forward(self, seq, target_item, dim):
        target_items = target_item.unsqueeze(-2).expand_as(seq)
        weights = self.attention_unit(target_items, seq)
        weights = torch.softmax(weights, dim=1)
        out = weights*seq
        return out.sum(dim=dim)


class Attention_Unit(nn.Module):

    def __init__(self, args):
        super(Attention_Unit, self).__init__()
        self.fc1 = nn.Linear(args.item_embed_size*4, args.item_embed_size)
        self.fc2 = nn.Linear(args.item_embed_size, 1)
        self.activation = torch.nn.ReLU()

    def forward(self, seq, target_item):
        emb_cat = torch.cat(
            (target_item, seq, target_item-seq, target_item*seq), dim=-1)
        x = self.activation(self.fc1(emb_cat))
        weight = self.fc2(x)
        return weight


def to_device(x, device):
    for key, value in x.items():
        x[key] = value.to(device)
    return x
