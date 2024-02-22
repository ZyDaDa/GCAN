import os
import argparse
from collections import defaultdict
import pickle
import math

if __name__=='__main__':
    # Dataset selection & File reading
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='RetailRocket', help='dataset name: diginetica/RetailRocket/Tmall')
    args = parser.parse_args()
    data_set = args.dataset
    data_folder = os.path.join(os.getcwd(), 'dataset',data_set)

    sess, tar = pickle.load(open(os.path.join(data_folder, 'train.txt'),'rb'))

    sessions = []
    for s,t in zip(sess,tar):
        sessions.append(s+[t])

    # Calculate similarity between items
    item_sim = {} 
    item_cnt = defaultdict(int) 
    for sess in sessions:
        session_weight = 1 / math.log(len(sess)+1) 
        for i in sess:
            item_cnt[i] += 1
            item_sim.setdefault(i,{})
            for j in sess:
                if i ==j:
                    continue
                item_sim[i].setdefault(j,0)
                item_sim[i][j] += session_weight 
                
    # Similarity Normalization
    item_sim_ = item_sim.copy()
    max_weigjt = 0.0
    for i, related_item in item_sim_.items():
        wight_max = 0.0
        for j, wij in related_item.items(): 
            now_weight = wij / math.sqrt(item_cnt[i]*item_cnt[j])
            item_sim_[i][j] = now_weight
            wight_max = max(wight_max, now_weight)
        for j, wij in related_item.items(): 
            item_sim_[i][j] = wij/wight_max
    
    pickle.dump(item_sim_, open(os.path.join(data_folder,'item_c_sim.pkl'),'wb'))