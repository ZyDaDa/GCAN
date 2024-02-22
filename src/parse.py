import argparse
import torch

def get_parse():

    parser = argparse.ArgumentParser()
    """ # diginetica
    parser.add_argument('--dataset', default='diginetica', help='diginetica/RetailRocket/Tmall/')
    parser.add_argument('--att_distance', type=int, default=4, help='attention distance')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')   
    parser.add_argument('--label_smoothing', type=float, default=0.15, help='label_smoothing')
    parser.add_argument('--layer_num', type=int, default=5, help='GA layer_num') 
    """
    """ # Tmall
    parser.add_argument('--dataset', default='Tmall', help='diginetica/RetailRocket/Tmall/')
    parser.add_argument('--att_distance', type=int, default=1, help='attention distance')
    parser.add_argument('--dropout', type=float, default=0.6, help='dropout rate')   
    parser.add_argument('--label_smoothing', type=float, default=0.2, help='label_smoothing')
    parser.add_argument('--layer_num', type=int, default=5, help='GA layer_num') 
    """
    """# RetailRocket"""
    parser.add_argument('--dataset', default='RetailRocket', help='diginetica/RetailRocket/Tmall/')
    parser.add_argument('--att_distance', type=int, default=8, help='attention distance')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')   
    parser.add_argument('--label_smoothing', type=float, default=0.15, help='label_smoothing')
    parser.add_argument('--layer_num', type=int, default=4, help='GA layer_num') 
    
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('--max_length', type=int, default=20, help='session max length')
    parser.add_argument('--heads', type=int, default=2, help='head num') 
    parser.add_argument('--dim', type=int, default=100, help='hidden state size')
    parser.add_argument('--epoch', type=int, default=10, help='the number of epochs to train for')
    parser.add_argument('--device', default='cuda', type=str,help='cuda or cpu')
    parser.add_argument('--topk', default=[10, 20], type=list)
    parser.add_argument('--lr', type=float, default=2e-3, help='learning rate') 
    parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
    parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
    parser.add_argument('--l2', type=float, default=0, help='l2 penalty') 
    
    args = parser.parse_args()
    if args.device == 'cuda':
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else: args.device = torch.device('cpu')
    return args
