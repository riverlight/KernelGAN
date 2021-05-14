import os
import tqdm

from configs import Config
from data import DataGenerator
from kernelGAN import KernelGAN
from learner import Learner
from torch.utils.data.dataloader import DataLoader


def train(conf):
    gan = KernelGAN(conf)
    learner = Learner()
    data = DataGenerator(conf, gan)
    train_dataloader = DataLoader(dataset=data,
                                  batch_size=8,
                                  shuffle=True,
                                  num_workers=8,
                                  pin_memory=False,
                                  drop_last=True)

    for iteration in tqdm.tqdm(range(conf.max_iters), ncols=60):
        for i, t_data in enumerate(train_dataloader):
            # [g_in, d_in] = data.__getitem__(iteration)
            g_in, d_in = t_data
            gan.train(g_in, d_in)
            learner.update(iteration, gan)
    gan.finish()


def main():
    """The main function - performs kernel estimation (+ ZSSR) for all images in the 'test_images' folder"""
    import argparse
    # Parse the command line arguments
    prog = argparse.ArgumentParser()
    prog.add_argument('--input-dir', '-i', type=str, default='test_images', help='path to image input directory.')
    prog.add_argument('--output-dir', '-o', type=str, default='results', help='path to image output directory.')
    prog.add_argument('--X4', action='store_true', help='The wanted SR scale factor')
    prog.add_argument('--SR', action='store_true', help='when activated - ZSSR is not performed')
    prog.add_argument('--real', action='store_true', help='ZSSRs configuration is for real images')
    prog.add_argument('--noise_scale', type=float, default=1., help='ZSSR uses this to partially de-noise images')
    args = prog.parse_args()
    # Run the KernelGAN sequentially on all images in the input directory
    flag = False
    for filename in os.listdir(os.path.abspath(args.input_dir)):
        print(filename)
        if filename == "0434.png":
            flag = True
        if flag is False:
            continue
        if os.path.isdir(os.path.join(args.input_dir, filename)):
            continue
        conf = Config().parse(create_params(filename, args))
        train(conf)
    prog.exit(0)


def create_params(filename, args):
    params = ['--input_image_path', os.path.join(args.input_dir, filename),
              '--output_dir_path', os.path.abspath(args.output_dir),
              '--noise_scale', str(args.noise_scale)]
    if args.X4:
        params.append('--X4')
    if args.SR:
        params.append('--do_ZSSR')
    if args.real:
        params.append('--real_image')
    return params


if __name__ == '__main__':
    main()
