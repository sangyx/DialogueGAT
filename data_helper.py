import torch
from torch.utils import data
import pickle
from functools import reduce
from fuzzywuzzy import fuzz


def get_data(data, year=2015):
    """
    2015         25/02/2015 - 22/10/2015(535/531)    22/10/2015 - 28/10/2015(76/75)     28/10/2015 - 17/12/2015(154/153)
    2016         05/01/2016 - 03/08/2016(980/968)    03/08/2016 - 12/08/2016(140/138)    15/08/2016 - 15/11/2016(280/278)
    2017-2018    17/01/2017 - 07/11/2017(894/890)    07/11/2017 - 15/02/2018(127/127)    15/02/2018 - 21/06/2018(257/255)
    """

    # split = {
    #     2015: ["20150225", "20151022", "20151028"],
    #     2016: ["20160105", "20160308", "20160815"],
    #     2017: ["20170117", "20171107", "20180215"],
    # }

    all_keys = []
    for key in data:
        date = key.split("_")[0]
        if year != 2017 and date[:4] != str(year):
            continue
        if year == 2017 and int(date[:4]) < year:
            continue

        all_keys.append(key)

    all_keys = sorted(all_keys)
    train_size, val_size = int(len(all_keys) * 0.7), int(len(all_keys) * 0.1)
    split_data = {
        "train": all_keys[:train_size],
        "val": all_keys[train_size : train_size + val_size],
        "test": all_keys[train_size + val_size :],
    }
    return split_data


class ECCDataset(data.Dataset):
    def __init__(self, ids, data, word2idx, max_len, target=3):
        self.target = target
        self.data = data
        self.ids = ids
        self.max_len = max_len
        self.word2idx = word2idx

    def pad(self, s):
        s_len = len(s)
        if s_len > self.max_len:
            s = s[: self.max_len]
            s_len = 256
        else:
            s = s + [0] * (self.max_len - s_len)
        return s

    def process_text(self, key):
        data = self.data[key]
        ps = []
        sen = []
        for t in data["transcript"]:
            if not t["speech"]:
                continue

            sp = reduce(lambda x, y: x + y, t["speech"])
            # sp = self.pad([self.word2idx[w] for w in sp if w in self.word2idx])
            sp = self.pad(sp)
            ps.append(t["name"])

            if not sp:
                continue

            sen.append(sp)

        return sen, ps

    def __getitem__(self, index):
        key = self.ids[index]
        sen, ps = self.process_text(key)
        label = self.data[key]["label"][self.target]
        plabel = self.data[key]["label"][-self.target]

        return key, sen, label, ps, len(sen), plabel

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    data = sorted(data, key=lambda x: x[4], reverse=True)
    keys = [d[0] for d in data]
    x = torch.cat([torch.tensor(d[1]) for d in data])
    y = torch.tensor([d[2] for d in data])
    ps = [d[3] for d in data]
    length = [d[4] for d in data]
    py = torch.tensor([d[5] for d in data])

    return keys, x, y, ps, length, py


if __name__ == "__main__":
    with open("../data/data.pkl", "rb") as f:
        all_data, vocab, word2idx, W, max_p_len, max_sen_len = pickle.load(f)

    split_data = get_data(all_data, year=2015)

    train_set = ECCDataset(split_data["train"], all_data, word2idx, max_p_len, target=3)
    val_set = ECCDataset(split_data["val"], all_data, word2idx, max_p_len, target=3)
    test_set = ECCDataset(split_data["test"], all_data, word2idx, max_p_len, target=3)

    train_dataloader = data.DataLoader(
        train_set, batch_size=64, shuffle=True, collate_fn=collate_fn
    )
    for keys, x, y, ps, length in train_dataloader:
        print(keys, x, y, ps, length)
