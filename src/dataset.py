import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import pickle
from tqdm import tqdm
import networkx as nx

def load_data(args):
    dataset_folder = os.path.abspath(os.path.join('dataset',args.dataset))
    train_set = SeqDataset(dataset_folder,'train', max_length=args.max_length)
    test_set = SeqDataset(dataset_folder,'test', max_length=args.max_length)
    train_loader = DataLoader(train_set,args.batch_size,  num_workers=5,
                              shuffle=True,collate_fn=collate_fn,drop_last=True)
    test_loader = DataLoader(test_set,args.batch_size, num_workers=0,
                              shuffle=False,collate_fn=collate_fn)
    if args.dataset == 'diginetica':
        item_num = 43097
    elif args.dataset == 'Tmall':
        item_num = 40728
    elif args.dataset == 'RetailRocket':
        item_num = 36968  
    else:
        pass
    return train_loader, test_loader, item_num

class SeqDataset(Dataset):
    def __init__(self, datafolder, file='train',max_length=19, augment=True) -> None:
        super().__init__()
        assert isinstance(max_length,int) and max_length > 0

        data_file = os.path.join(datafolder, file+'.txt')
        self.max_length = max_length
        processed_file = os.path.join(os.path.join(datafolder, file+'_maxlen_%d.pkl'%(max_length)))

        if os.path.exists(processed_file):
            self.all_info =  pickle.load(open(processed_file, 'rb'))   
        else:
            self.item_sim = pickle.load(open(os.path.join(datafolder, 'item_c_sim.pkl'),'rb')) 

            self.all_info = [] 

            data_file = os.path.join(datafolder, file+'.txt')
            sessions_pkl = pickle.load(open(data_file, 'rb'))   
            sessions, target = sessions_pkl 
            all_sess = []
            for s,t in zip(sessions, target):
                all_sess.append(s+[t])

            # Calculate the required data and save it
            for sess in tqdm(all_sess, ncols=80, desc=file):
                all_info = self.__cons_sess_info__(sess)
                self.all_info.append(all_info)
    
            print("Session number is %d"%(len(self.all_info)))
            pickle.dump((self.all_info), open(processed_file,'wb'))

    def __cons_sess_info__(self, sess):
        # Build additional information needed from session data
        data = sess[-self.max_length-1:] 

        session = data[:-1] 
        tar = data[-1] 

        spd = []
        sim = [] 
        # get Shortest Path in Session Graph
        G = nx.Graph()
        e_list = [(session[idx], session[idx+1]) for idx in range(len(session)-1)] 
        G.add_edges_from(e_list)    
        p = nx.shortest_path(G)

        for idx, v in enumerate(session):
            if len(session) == 1: 
                node_spd = [1] 
            else:
                node_spd = [len(p[v][t]) for t in session] 
            spd.append(node_spd) 
            node_sim = [self.item_sim.get(t,{}).get(v,0) for t in session]
            sim.append(node_sim)
        session_pad = [0]*(self.max_length-len(session)) + session
        spd_pad = [[0]*self.max_length]*(self.max_length-len(session)) 
        sim_pad = [[0]*self.max_length]*(self.max_length-len(session)) 
        for s,si in zip(spd,sim):
            spd_pad.append([0]*(self.max_length-len(s))+s)
            sim_pad.append([0]*(self.max_length-len(si))+si)
        mask = [True]*(self.max_length-len(session)) + [False]*len(session)

        # session squence, label, Shortest Path Distance Matrix, Collaborative Similarity Matrix, mask
        return session_pad, tar, spd_pad, sim_pad, mask

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, index): 
        session_pad, tar, spd_pad, sim_pad, mask = self.all_info[index]
        return session_pad, tar, spd_pad, sim_pad, mask

def collate_fn(batch_data):
    data_list = []
    label_list = []
    spd_list = []
    sim_list = []
    mask_list = []

    for data in batch_data:
        x = data[0]
        y = data[1]
        spd = data[2]
        sim = data[3]
        mask = data[4]

        data_list.append(x)
        label_list.append(y)
        spd_list.append(spd)
        sim_list.append(sim)
        mask_list.append(mask)

    data = torch.LongTensor(data_list)
    label = torch.LongTensor(label_list)
    spd = torch.LongTensor(spd_list)
    sim = torch.FloatTensor(sim_list)
    mask = torch.BoolTensor(mask_list)
    return data, spd, sim, mask, label