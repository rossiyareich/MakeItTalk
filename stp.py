import argparse
import glob
import os
import pickle
import sys

sys.path.append('MakeItTalk')
sys.path.append('MakeItTalk/thirdparty/AdaptiveWingLoss')

import cv2
import face_alignment
import numpy as np
import torch
import util.utils as util
from scipy.signal import savgol_filter
from src.approaches.train_audio2landmark import Audio2landmark_model
from src.approaches.train_image_translation import Image_translation_block
from src.autovc.AutoVC_mel_Convertor_retrain_version import \
    AutoVC_mel_Convertor


DEBUG = False


class STP:
    def __init__(self, head_name='scarlett', naive_eye=False, is_mouth_open=False):
        self.native_cwd = os.getcwd()
        os.chdir('MakeItTalk')

        # Step 1: Basic setup for the animation
        # the image name (with no .jpg) to animate
        default_head_name = head_name
        self.ADD_NAIVE_EYE = naive_eye          # whether add naive eye blink
        # if your image has an opened mouth, put this as True, else False
        CLOSE_INPUT_FACE_MOUTH = is_mouth_open

        parser = argparse.ArgumentParser()
        parser.add_argument('--jpg', type=str,
                            default='{}.jpg'.format(default_head_name))
        parser.add_argument('--close_input_face_mouth',
                            default=CLOSE_INPUT_FACE_MOUTH, action='store_true')

        parser.add_argument('--load_AUTOVC_name', type=str,
                            default='examples/ckpt/ckpt_autovc.pth')
        parser.add_argument('--load_a2l_G_name', type=str,
                            default='examples/ckpt/ckpt_speaker_branch.pth')
        # ckpt_audio2landmark_c.pth')
        parser.add_argument('--load_a2l_C_name', type=str,
                            default='examples/ckpt/ckpt_content_branch.pth')
        # ckpt_image2image.pth') #ckpt_i2i_finetune_150.pth') #c
        parser.add_argument('--load_G_name', type=str,
                            default='examples/ckpt/ckpt_116_i2i_comb.pth')

        parser.add_argument('--amp_lip_x', type=float, default=2.)
        parser.add_argument('--amp_lip_y', type=float, default=2.)
        parser.add_argument('--amp_pos', type=float, default=.5)
        # ['iWeklsXc0H8']) #['45hn7-LXDX8']) #['E_kmpT-EfOg']) #'iWeklsXc0H8', '29k8RtSUjE0', '45hn7-LXDX8',
        parser.add_argument('--reuse_train_emb_list',
                            type=str, nargs='+', default=[])
        parser.add_argument('--add_audio_in', default=False,
                            action='store_true')
        parser.add_argument('--comb_fan_awing',
                            default=False, action='store_true')
        parser.add_argument('--output_folder', type=str, default='examples')

        parser.add_argument('--test_end2end', default=True,
                            action='store_true')
        parser.add_argument('--dump_dir', type=str, default='', help='')
        parser.add_argument('--pos_dim', default=7, type=int)
        parser.add_argument('--use_prior_net',
                            default=True, action='store_true')
        parser.add_argument('--transformer_d_model', default=32, type=int)
        parser.add_argument('--transformer_N', default=2, type=int)
        parser.add_argument('--transformer_heads', default=2, type=int)
        parser.add_argument('--spk_emb_enc_size', default=16, type=int)
        parser.add_argument('--init_content_encoder', type=str, default='')
        parser.add_argument('--lr', type=float,
                            default=1e-3, help='learning rate')
        parser.add_argument('--reg_lr', type=float,
                            default=1e-6, help='weight decay')
        parser.add_argument('--write', default=False, action='store_true')
        parser.add_argument('--segment_batch_size', type=int,
                            default=1, help='batch size')
        parser.add_argument('--emb_coef', default=3.0, type=float)
        parser.add_argument('--lambda_laplacian_smooth_loss',
                            default=1.0, type=float)
        parser.add_argument('--use_11spk_only',
                            default=False, action='store_true')

        self.opt_parser = parser.parse_args()

        # Step 2: Load the image and detect its landmark
        self.img = cv2.imread('examples/' + self.opt_parser.jpg)
        predictor = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._3D, device='cuda', flip_input=True)
        shapes = predictor.get_landmarks(self.img)
        if (not shapes or len(shapes) != 1):
            print('Cannot detect face landmarks. Exit.')
            exit(-1)
        self.shape_3d = shapes[0]

        if(self.opt_parser.close_input_face_mouth):
            util.close_input_face_mouth(self.shape_3d)

        # (Optional) Simple manual adjustment to landmarks in case FAN is not accurate, e.g.
        self.shape_3d[48:, 0] = (self.shape_3d[48:, 0] - np.mean(
            self.shape_3d[48:, 0])) * 1.05 + np.mean(self.shape_3d[48:, 0])  # wider lips
        self.shape_3d[49:54, 1] += 0.           # thinner upper lip
        self.shape_3d[55:60, 1] -= 1.           # thinner lower lip
        self.shape_3d[[37, 38, 43, 44], 1] -= 2.    # larger eyes
        self.shape_3d[[40, 41, 46, 47], 1] += 2.    # larger eyes

        # Normalize face as input to audio branch
        self.shape_3d, self.scale, self.shift = util.norm_input_face(
            self.shape_3d)
        os.chdir(self.native_cwd)

    def generate(self, audio_filepath, output_filepath):
        os.chdir('MakeItTalk')
        # Step 3: Generate input data for inference based on uploaded audio
        au_data = []
        au_emb = []

        os.system(
            f'ffmpeg -y -loglevel error -i ../{audio_filepath} -ar 16000 examples/tmp.wav')

        # au embedding
        from thirdparty.resemblyer_util.speaker_emb import get_spk_emb
        me, ae = get_spk_emb('examples/tmp.wav')
        au_emb.append(me.reshape(-1))

        if DEBUG:
            print('Processing audio file')
        c = AutoVC_mel_Convertor('examples')
        au_data_i = c.convert_single_wav_to_autovc_input(
            audio_filename='examples/tmp.wav', autovc_model_path=self.opt_parser.load_AUTOVC_name)
        au_data += au_data_i
        if(os.path.isfile('examples/tmp.wav')):
            os.remove('examples/tmp.wav')
        if DEBUG:
            print("Loaded audio...", file=sys.stderr)

        # landmark fake placeholder
        fl_data = []
        rot_tran, rot_quat, anchor_t_shape = [], [], []
        for au, info in au_data:
            au_length = au.shape[0]
            fl = np.zeros(shape=(au_length, 68 * 3))
            fl_data.append((fl, info))
            rot_tran.append(np.zeros(shape=(au_length, 3, 4)))
            rot_quat.append(np.zeros(shape=(au_length, 4)))
            anchor_t_shape.append(np.zeros(shape=(au_length, 68 * 3)))

        if(os.path.exists(os.path.join('examples', 'dump', 'random_val_fl.pickle'))):
            os.remove(os.path.join('examples', 'dump', 'random_val_fl.pickle'))
        if(os.path.exists(os.path.join('examples', 'dump', 'random_val_fl_interp.pickle'))):
            os.remove(os.path.join('examples', 'dump',
                      'random_val_fl_interp.pickle'))
        if(os.path.exists(os.path.join('examples', 'dump', 'random_val_au.pickle'))):
            os.remove(os.path.join('examples', 'dump', 'random_val_au.pickle'))
        if (os.path.exists(os.path.join('examples', 'dump', 'random_val_gaze.pickle'))):
            os.remove(os.path.join('examples', 'dump',
                      'random_val_gaze.pickle'))

        with open(os.path.join('examples', 'dump', 'random_val_fl.pickle'), 'wb') as fp:
            pickle.dump(fl_data, fp)
        with open(os.path.join('examples', 'dump', 'random_val_au.pickle'), 'wb') as fp:
            pickle.dump(au_data, fp)
        with open(os.path.join('examples', 'dump', 'random_val_gaze.pickle'), 'wb') as fp:
            gaze = {'rot_trans': rot_tran, 'rot_quat': rot_quat,
                    'anchor_t_shape': anchor_t_shape}
            pickle.dump(gaze, fp)

        # Step 4: Audio-to-Landmarks prediction
        model = Audio2landmark_model(self.opt_parser, jpg_shape=self.shape_3d)
        if(len(self.opt_parser.reuse_train_emb_list) == 0):
            model.test(au_emb=au_emb, vis_fls=False)
        else:
            model.test(au_emb=None, vis_fls=False)
        if DEBUG:
            print("Audio->Landmark...", file=sys.stderr)

        # Step 5: Natural face animation via Image-to-image translation
        fls = glob.glob1('examples', 'pred_fls_*.txt')
        fls.sort()

        for i in range(0, len(fls)):
            fl = np.loadtxt(os.path.join(
                'examples', fls[i])).reshape((-1, 68, 3))
            fl[:, :, 0:2] = -fl[:, :, 0:2]
            fl[:, :, 0:2] = fl[:, :, 0:2] / self.scale - self.shift

            if (self.ADD_NAIVE_EYE):
                fl = util.add_naive_eye(fl)

            # additional smooth
            fl = fl.reshape((-1, 204))
            fl[:, :48 * 3] = savgol_filter(fl[:, :48 * 3], 15, 3, axis=0)
            fl[:, 48*3:] = savgol_filter(fl[:, 48*3:], 5, 3, axis=0)
            fl = fl.reshape((-1, 68, 3))

            # Step 6: Image2image translation
            model = Image_translation_block(self.opt_parser, single_test=True)
            with torch.no_grad():
                model.single_test(jpg=self.img, fls=fl,
                                  out_file=output_filepath)
                if DEBUG:
                    print('finish image2image gen')
            os.remove(os.path.join('examples', fls[i]))
            if DEBUG:
                print("{} / {}: Landmark->Face...".format(i +
                      1, len(fls)), file=sys.stderr)
        if DEBUG:
            print("Done!", file=sys.stderr)
        os.chdir(self.native_cwd)
