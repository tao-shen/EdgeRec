from utils import *
from model import *
# random seed
setup_seed(2021)
# initialize hyperparameters
args = init_args()
# initialize model
model = DIN(args)
# record loss & auc
recorder = {'loss': [], 'auc': []}
# load csv-fotmat dataset
train_set = MCC_Dataset('trainset_{}.csv'.format(args.datasize), args)
test_set = MCC_Dataset('testset_{}.csv'.format(args.datasize), args)
# dataloader & optimizer
train_loader = data_loader(train_set, batchsize=args.batchsize)
test_loader = data_loader(test_set, batchsize=args.batchsize)
optimizer = torch.optim.Adam(
    model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if __name__ == '__main__':
    # start training % testing
    for _ in range(10):
        model.fit(train_loader, optimizer)
        model.evaluate(test_loader, recorder)
        print(recorder['auc'][-1])
