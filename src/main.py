from tqdm import tqdm
from dataset import load_data
import torch
from parse import get_parse
from utils import fix_seed, topk_mrr_hr
from GCAN.GCAN import GCAN
import time


def main(seed=42):

    fix_seed(seed)
    args = get_parse()

    train_loader, test_loader, num_items = load_data(args)
    model = GCAN(args, num_items)
    model.to(args.device)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.l2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)

    mini_batch = 0
    hr_max, mrr_max = {k:0 for k in args.topk}, {k:0 for k in args.topk} 
    for e in range(args.epoch):
        model.train()
        all_loss = 0.0

        bar = tqdm(train_loader, total=len(train_loader),ncols=100)

        for data, spd, sim, mask, tar in bar:
            
            output = model(data.to(args.device), 
                            spd.to(args.device),
                            sim.to(args.device),
                            mask.to(args.device))
            scores = model.comp_scores(output, mask.to(args.device))
            optimizer.zero_grad() 
            loss = model.loss_function(scores, tar.to(args.device)-1) 
            loss.backward()
            optimizer.step()
            all_loss += loss.item()
            mini_batch += 1
            bar.set_postfix(Epoch=e, LR=optimizer.param_groups[0]['lr'], Train_Loss=loss.item()/data.size(0))
        scheduler.step()

        print('epoch%d - loss%f'%(e,all_loss/len(train_loader)))
        
        model.eval()
        all_loss = 0.0
        hr, mrr = dict((k,0.0) for k in args.topk), dict((k,0.0) for k in args.topk) 
        test_num = 0
        st_time = time.time()
        for data, spd, sim, mask, tar in tqdm(test_loader,ncols=80,desc='test'):
            output = model(data.to(args.device), 
                            spd.to(args.device),
                            sim.to(args.device),
                            mask.to(args.device))
            scores = model.comp_scores(output, mask.to(args.device))
            loss = model.loss_function(scores, tar.to(args.device)-1) 
            all_loss += loss.item()
            for k in hr.keys():
                this_hr, this_mrr = topk_mrr_hr(scores.detach().cpu(),(tar-1).numpy(),k)
                hr[k] += this_hr
                mrr[k] += this_mrr
            test_num += data.shape[0]

        print("Test time %.2f"%(time.time()-st_time))
        for k in hr.keys():
            hr[k] /= test_num
            mrr[k] /= test_num
 
            if hr_max[k] < hr[k]: hr_max[k] = hr[k]
            if mrr_max[k] < mrr[k]: mrr_max[k] = mrr[k]

            print("best HR@%d %.2f\tMRR@%d %.2f"%(k,hr_max[k]*100,k,mrr_max[k]*100))
            print("now  HR@%d %.2f\tMRR@%d %.2f"%(k,hr[k]*100,k,mrr[k]*100))
            
    # Print the best score
    for k in args.topk:
        print('Top%d\thit%.2f\tmrr%.2f\tbest'%(k,hr_max[k]*100,mrr_max[k]*100))


if __name__ == '__main__':
    main()

        
