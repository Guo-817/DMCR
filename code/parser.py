import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run DMCR.")

    parser.add_argument('--dataset', nargs='?', default='RateBeer',
                        help='Choose a dataset from {Yahoo, TripAdvisor, RateBeer, Yelp2022, BeerAdvocate, RB-Extended}')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')

    parser.add_argument('--Ks', nargs='?', default='[20, 50]',
                        help='K for Top-K list')

    parser.add_argument('--embed_size', type=int, default=64,   #common parameter
                        help='Embedding size.')
    parser.add_argument('--layer_size', nargs='?', default='[64,64,64,64]', #common parameter
                        help='Output sizes of every layer')

    parser.add_argument('--coefficient', nargs='?', default='[0.4, 0.4, 0.4, 0.8, 0.8, 0.8, 0.8, 1.0]',
                        help='Regularization')

    parser.add_argument('--epoch', type=int, default=500,
                        help='Number of epoch')

    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')

    parser.add_argument('--regs', nargs='?', default='[1e-5,1e-5,1e-2]',
                        help='Regularizations')

    parser.add_argument('--stopping_step_flag', type=int, default=10,
                        help='stopping_step')

    parser.add_argument('--test_epoch', type=int, default=5,
                        help='test epoch steps')

    parser.add_argument('--weights_path', nargs='?', default='',
                        help='Store model path')
    parser.add_argument('--data_path', nargs='?', default='../Data/',
                        help='Input data path')
    parser.add_argument('--proj_path', nargs='?', default='',
                        help='Project path')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation')
    parser.add_argument('--is_norm', type=int, default=1,
                    help='Interval of evaluation')

    parser.add_argument('--nhead', type=int, default=2)

    parser.add_argument('--adj_type', nargs='?', default='pre',
                        help='Specify the type of the adjacency (laplacian) matrix from {plain, norm, mean}')
    parser.add_argument('--gpu_id', type=int, default=2,
                        help='Gpu id')

    parser.add_argument('--node_dropout_flag', type=int, default=1,
                        help='0: Disable node dropout, 1: Activate node dropout')

    parser.add_argument('--node_dropout', nargs='?', default='[0.1]',
                        help='Keep probability')

    parser.add_argument('--save_flag', type=int, default=0,
                        help='0: Disable model saver, 1: Activate model saver')

    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}')

    parser.add_argument('--wid', nargs='?', default='[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]',
                        help='negative weight')

    parser.add_argument('--att_dim', type=int, default=16, help='self att dim')

    parser.add_argument('--decay', type=float, default=10,
                        help='Regularization')

    parser.add_argument('--mess_dropout', nargs='?', default='[0.3]',
                        help='Keep probability w.r.t. message dropout')

    return parser.parse_args()
