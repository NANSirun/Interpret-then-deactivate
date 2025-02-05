import argparse
from cleanfid import fid

def main(args):
    if args.fid:
        fid_value = fid.compute_fid(args.dir1, args.dir2, device=args.device)
        print(f'FID score: {fid_value}')
    else:
        fid_value=None
    if args.kid:
        kid_value = fid.compute_kid(args.dir1, args.dir2, device=args.device)
        print(f'KID score: {kid_value}')
    else:
        kid_value=None
    
    if not args.fid and not args.kid:
        print("Please specify at least one metric to compute using --fid or --kid.")
    
    with open("fid_kid.log", "a+") as f:
        f.write('\n' + '=' * 30 + '\n')
        f.write(f'Arguments: {args}\n')
        f.write(f'Dir1: {args.dir1}\n')
        f.write(f'Dir2: {args.dir2}\n')
        if fid_value is not None:
            f.write(f'FID: {fid_value:.5f}\n')
        if kid_value is not None:
            f.write(f'KID: {kid_value:.5f}\n')
        f.write('=' * 30 + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute FID score between two directories.')
    parser.add_argument('--dir1', type=str, required=True, help='Path to the first directory')
    parser.add_argument('--dir2', type=str, required=True, help='Path to the second directory')
    parser.add_argument('--fid', action='store_true', help='Compute the FID score.')
    parser.add_argument('--kid', action='store_true', help='Compute the KID score.')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    main(args)