import argparse


def construct_model_id(model, gpt_layers, seq_len, pred_len, percent):
    """构建 model_id 字符串"""
    return f"ETTh1_{model}_{gpt_layers}_{seq_len}_{pred_len}_{percent}"


def get_args():
    parser = argparse.ArgumentParser(description='GPT4TS')

    # 基础路径和数据参数
    parser.add_argument('--model_id', type=str, default='test')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/',help="数据集文件")
    parser.add_argument('--root_path', type=str, default='./datasets/ETT-small/', help="数据集所在的文件夹")
    parser.add_argument('--data_path', type=str, default='ETTh1.csv')
    parser.add_argument('--data', type=str, default='ett_h')  # 指明要用那个数据集
    parser.add_argument('--features', type=str, default='M', help="指明时间序列是否为多变量序列")

    # 来自 bash 的参数设置
    parser.add_argument('--seq_len', type=int, default=336,help="输入序列的长度")  # bash: seq_len=336
    parser.add_argument('--model', type=str, default='GPT4TS')  # bash: model=GPT4TS
    parser.add_argument('--pred_len', type=int, default=96,help="未来预测的长度")  # bash: pred_len in [96, 192, 336, 720]
    parser.add_argument('--label_len', type=int, default=168,help="预测已知序列的一部分")  # bash: label_len=168
    parser.add_argument('--batch_size', type=int, default=256)  # bash: batch_size=256
    parser.add_argument('--learning_rate', type=float, default=0.0001)  # bash: lr=0.0001
    parser.add_argument('--train_epochs', type=int, default=10)  # bash: train_epochs=10
    parser.add_argument('--decay_fac', type=float, default=0.5)  # bash: decay_fac=0.5
    parser.add_argument('--d_model', type=int, default=768)  # bash: d_model=768
    parser.add_argument('--n_heads', type=int, default=4)  # bash: n_heads=4
    parser.add_argument('--d_ff', type=int, default=768)  # bash: d_ff=768
    parser.add_argument('--freq', type=int, default=0)  # bash: freq=0
    parser.add_argument('--patch_size', type=int, default=16,help="patch大小")  # bash: patch_size=16
    parser.add_argument('--stride', type=int, default=8)  # bash: stride=8
    parser.add_argument('--percent', type=int, default=100, help="采用部分样本就行训练时可以用到")  # bash: percent=100
    parser.add_argument('--gpt_layers', type=int, default=6,help="GPT的层数")  # bash: gpt_layer=6
    parser.add_argument('--itr', type=int, default=3)  # bash: itr=3
    parser.add_argument('--cos', type=int, default=1)  # bash: cos=1
    parser.add_argument('--tmax', type=int, default=20)  # bash: tmax=20
    parser.add_argument('--is_gpt', type=int, default=1)  # bash: is_gpt=1

    # 其他默认参数
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--embed', type=str, default='timeF',help="是否使用预定义的时间特征提取函数")
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--lradj', type=str, default='type4')  # 修改为 type4
    parser.add_argument('--e_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.3)  # 修改为 0.3
    parser.add_argument('--enc_in', type=int, default=7)  # 修改为 7
    parser.add_argument('--c_out', type=int, default=7)  # 修改为 7
    parser.add_argument('--kernel_size', type=int, default=25)
    parser.add_argument('--loss_func', type=str, default='mse')
    parser.add_argument('--pretrain', type=int, default=1,help="是否采用预训练模型")
    parser.add_argument('--freeze', type=int, default=1)
    parser.add_argument('--max_len', type=int, default=-1)
    parser.add_argument('--hid_dim', type=int, default=16)
    parser.add_argument('--patience', type=int, default=3)

    args = parser.parse_args()

    # 确保 model_id 总是基于最新的参数值
    args.model_id = construct_model_id(
        args.model,
        args.gpt_layers,
        args.seq_len,
        args.pred_len,
        args.percent
    )

    return args


def print_args(args):
    """打印所有参数值"""
    print("\n=== 模型参数设置 ===")
    print(f"Model ID: {args.model_id}")  # 特别突出显示 model_id
    print("\n其他参数:")
    for arg in vars(args):
        if arg != 'model_id':  # model_id 已经打印过了
            print(f"{arg}: {getattr(args, arg)}")
    print("==================\n")
