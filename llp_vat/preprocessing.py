import argparse

from llp_vat.lib.llp import create_llp_dataset


def main(args):
    # create LLP dataset
    if args.alg == "uniform":
        kwargs = dict(replacement=args.replacement,
                      bag_size=args.bag_size,
                      seed=args.seed)
    elif args.alg == "kmeans":
        kwargs = dict(n_clusters=args.n_clusters,
                      reduction=args.reduction,
                      seed=args.seed)
    else:
        raise NameError("The bag creation algorithm is not supported")
    create_llp_dataset(args.domain_index,
                       args.obj_dir,
                       args.alg,
                       **kwargs)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_dir", default="./obj")
    parser.add_argument("-i", "--domain_index", type=int, required=True, default=10)
    parser.add_argument("--alg", choices=["uniform", "kmeans"], required=True, default='uniform')
    parser.add_argument("-b", "--bag_size", type=int, default=64)
    parser.add_argument("--replacement", action="store_true")
    parser.add_argument("-k", "--n_clusters", type=int)
    parser.add_argument("--reduction", type=int, default=600)
    parser.add_argument("--seed", default=0, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
