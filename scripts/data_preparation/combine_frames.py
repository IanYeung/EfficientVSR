import os
import cv2
import glob


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def encode_video_with_ffmpeg(src_path, dst_path, fps=25, crf=0, lib='libx264'):
    command = 'ffmpeg -r {} -f image2 -i {} -c:v {} -crf {} -pix_fmt yuv420p -an {} -y'\
        .format(fps, src_path, lib, crf, dst_path)
    print(command)
    os.system(command)


def encode_video_with_opencv(folder_path, output_path, fps=25):
    frame_paths = sorted(glob.glob(os.path.join(folder_path, '*.png')))

    h, w = 720, 1280
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame_path in frame_paths:
        print(frame_path)
        frame = cv2.imread(frame_path)
        video_writer.write(frame)

    cv2.destroyAllWindows()
    video_writer.release()


def encode_sr():
    expn = f'BasicVSRGAN_UNetDiscriminatorWithSpectralNorm_REDS_BDx4_LDL5e2'
    root = f'/home/xiyang/Results/BasicSR/results/{expn}'
    dataset = 'REDS4'

    read_root = os.path.join(root, 'visualization')
    save_root = os.path.join(root, 'video')

    mkdir(os.path.join(save_root, dataset))

    seqlist = [os.path.basename(x) for x in sorted(glob.glob(os.path.join(read_root, dataset, '*')))]

    for seqname in seqlist:
        print(f'Processing {seqname} ...')
        src_path = os.path.join(read_root, dataset, seqname, f'%08d_{expn}.png')
        dst_path = os.path.join(save_root, dataset, f'{seqname}.mp4')
        encode_video_with_ffmpeg(src_path=src_path, dst_path=dst_path, fps=25, crf=0)


def encode_gt():
    mode = 'test_sharp'
    read_root = f'/home/xiyang/Datasets/VSR-TEST/REDS4/{mode}'
    save_root = f'/home/xiyang/Datasets/VSR-TEST/REDS4-video/{mode}'

    mkdir(save_root)

    seqlist = [os.path.basename(x) for x in sorted(glob.glob(os.path.join(read_root, '*')))]

    for seqname in seqlist:
        print(f'Processing {seqname} ...')
        src_path = os.path.join(read_root, seqname, '%08d.png')
        dst_path = os.path.join(save_root, f'{seqname}.mp4')
        encode_video_with_ffmpeg(src_path=src_path, dst_path=dst_path, fps=25, crf=0)


if __name__ == '__main__':
    encode_sr()
