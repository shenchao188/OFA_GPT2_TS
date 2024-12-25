from data_provider.data_loader import Dataset_Custom, Dataset_Pred, Dataset_TSF, Dataset_ETT_hour, Dataset_ETT_minute
from torch.utils.data import DataLoader

data_dict = {
    'custom': Dataset_Custom,
    'tsf_data': Dataset_TSF,
    'ett_h': Dataset_ETT_hour,
    'ett_m': Dataset_ETT_minute,
}


def data_provider(args, flag, drop_last_test=True, train_all=False):
    Data = data_dict[args.data]  # 选择需要使用的数据集加载器
    timeenc = 0 if args.embed != 'timeF' else 1  # 是否使用预定义的时间特征提取函数
    percent = args.percent  # 采用部分样本就行训练时可以用到
    max_len = args.max_len

    if flag == 'test':
        shuffle_flag = False
        drop_last = drop_last_test
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    elif flag == 'val':
        shuffle_flag = True
        drop_last = drop_last_test
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    # 上面都是关于数据处理的一些参数设置

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        percent=percent,
        max_len=max_len,
        train_all=train_all
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,  # 训练集的时候会打乱来处理
        num_workers=args.num_workers,
        drop_last=drop_last)  # 通过访问__getitem__(self, index)函数来划分batch
    return data_set, data_loader
