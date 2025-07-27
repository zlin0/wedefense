#!/usr/bin/env python3
# The original version is from https://github.com/TakHemlata/RawBoost-antispoofing/blob/main/data_utils_rawboost.py  # noqa
# For paper: RawBoost: A Raw Data Boosting and Augmentation Method applied to
#                      Automatic Speaker Verification Anti-Spoofing
# https://arxiv.org/pdf/2111.04433
# Author: Hemlata Tak, Madhu Kamble, Jose Patino, Massimiliano Todisco and Nicholas Evans  # noqa

import argparse
import scipy.io.wavfile as sciwav
from wedefense.dataset.augmentation.rawboost import ISD_additive_noise, LnL_convolutive_noise, SSI_additive_noise, normWav

# --------------RawBoost data augmentation algorithms------------------------##


def process_Rawboost_feature(feature, sr, args, algo):

    # Data process by Convolutive noise (1st algo)
    if algo == 1:

        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands,
                                        args.minF, args.maxF, args.minBW,
                                        args.maxBW, args.minCoeff,
                                        args.maxCoeff, args.minG, args.maxG,
                                        args.minBiasLinNonLin,
                                        args.maxBiasLinNonLin, sr)

    # Data process by Impulsive noise (2nd algo)
    elif algo == 2:

        feature = ISD_additive_noise(feature, args.P, args.g_sd)

    # Data process by coloured additive noise (3rd algo)
    elif algo == 3:

        feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax,
                                     args.nBands, args.minF, args.maxF,
                                     args.minBW, args.maxBW, args.minCoeff,
                                     args.maxCoeff, args.minG, args.maxG, sr)

    # Data process by all 3 algo. together in series (1+2+3)
    elif algo == 4:

        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands,
                                        args.minF, args.maxF, args.minBW,
                                        args.maxBW, args.minCoeff,
                                        args.maxCoeff, args.minG, args.maxG,
                                        args.minBiasLinNonLin,
                                        args.maxBiasLinNonLin, sr)
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
        feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax,
                                     args.nBands, args.minF, args.maxF,
                                     args.minBW, args.maxBW, args.minCoeff,
                                     args.maxCoeff, args.minG, args.maxG, sr)

    # Data process by 1st two algo. together in series (1+2)
    elif algo == 5:

        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands,
                                        args.minF, args.maxF, args.minBW,
                                        args.maxBW, args.minCoeff,
                                        args.maxCoeff, args.minG, args.maxG,
                                        args.minBiasLinNonLin,
                                        args.maxBiasLinNonLin, sr)
        feature = ISD_additive_noise(feature, args.P, args.g_sd)

    # Data process by 1st and 3rd algo. together in series (1+3)
    elif algo == 6:

        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands,
                                        args.minF, args.maxF, args.minBW,
                                        args.maxBW, args.minCoeff,
                                        args.maxCoeff, args.minG, args.maxG,
                                        args.minBiasLinNonLin,
                                        args.maxBiasLinNonLin, sr)
        feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax,
                                     args.nBands, args.minF, args.maxF,
                                     args.minBW, args.maxBW, args.minCoeff,
                                     args.maxCoeff, args.minG, args.maxG, sr)

    # Data process by 2nd and 3rd algo. together in series (2+3)
    elif algo == 7:

        feature = ISD_additive_noise(feature, args.P, args.g_sd)
        feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax,
                                     args.nBands, args.minF, args.maxF,
                                     args.minBW, args.maxBW, args.minCoeff,
                                     args.maxCoeff, args.minG, args.maxG, sr)

    # Data process by 1st two algo. together in Parallel (1||2)
    elif algo == 8:

        feature1 = LnL_convolutive_noise(feature, args.N_f, args.nBands,
                                         args.minF, args.maxF, args.minBW,
                                         args.maxBW, args.minCoeff,
                                         args.maxCoeff, args.minG, args.maxG,
                                         args.minBiasLinNonLin,
                                         args.maxBiasLinNonLin, sr)
        feature2 = ISD_additive_noise(feature, args.P, args.g_sd)

        feature_para = feature1 + feature2
        feature = normWav(feature_para, 0)  # normalized resultant waveform

    # original data without Rawboost processing
    else:

        feature = feature

    return feature


# --------------Parameters for RawBoost data augmentation algorithms--------- ##


# Raeboost contain too many parameters, and we usually use the default configuration.  # noqa
# So to make life easier, I put parameters here.
# But need to #TODO pay attention for sampling rate related parameters.
def get_args_for_rawboost():
    parser = argparse.ArgumentParser(add_help=False)

    # LnL_convolutive_noise parameters
    parser.add_argument(
        '--nBands',
        type=int,
        default=5,
        help='number of notch filters.The higher the number of bands, '
        'the more aggresive the distortions is.[default=5]')
    parser.add_argument(
        '--minF',
        type=int,
        default=20,
        help='minimum centre frequency [Hz] of notch filter.[default=20]')
    parser.add_argument(
        '--maxF',
        type=int,
        default=8000,
        help='maximum centre frequency [Hz] (<sr/2) of notch filter. '
        '[default=8000]')
    parser.add_argument('--minBW',
                        type=int,
                        default=100,
                        help='minimum width [Hz] of filter.[default=100]')
    parser.add_argument('--maxBW',
                        type=int,
                        default=1000,
                        help='maximum width [Hz] of filter.[default=1000]')
    parser.add_argument(
        '--minCoeff',
        type=int,
        default=10,
        help='minimum filter coefficients. '
        'More the filter coefficients more ideal the filter slope. '
        '[default=10]')
    parser.add_argument(
        '--maxCoeff',
        type=int,
        default=100,
        help='maximum filter coefficients. '
        'More the filter coefficients more ideal the filter slope. '
        '[default=100]')
    parser.add_argument(
        '--minG',
        type=int,
        default=0,
        help='minimum gain factor of linear component.[default=0]')
    parser.add_argument(
        '--maxG',
        type=int,
        default=0,
        help='maximum gain factor of linear component.[default=0]')
    parser.add_argument(
        '--minBiasLinNonLin',
        type=int,
        default=5,
        help='minimum gain difference between linear and non-linear components. '
        '[default=5]')
    parser.add_argument(
        '--maxBiasLinNonLin',
        type=int,
        default=20,
        help='maximum gain difference between linear and non-linear components. '
        '[default=20]')
    parser.add_argument('--N_f',
                        type=int,
                        default=5,
                        help='order of the (non-)linearity component. '
                        'where N_f=1 refers only to linear components. '
                        '[default=5]')

    # ISD_additive_noise parameters
    parser.add_argument(
        '--P',
        type=int,
        default=10,
        help='Maximum number of uniformly distributed samples in [%]. '
        '[defaul=10]')
    parser.add_argument('--g_sd',
                        type=int,
                        default=2,
                        help='gain parameters > 0. [default=2]')

    # SSI_additive_noise parameters
    parser.add_argument(
        '--SNRmin',
        type=int,
        default=10,
        help='Minimum SNR value for coloured additive noise.[defaul=10]')
    parser.add_argument(
        '--SNRmax',
        type=int,
        default=40,
        help='Maximum SNR value for coloured additive noise.[defaul=40]')

    # ================== Rawboost data augmentation ========================== #
    args, _ = parser.parse_known_args()

    return args


def debug():

    args = get_args_for_rawboost()
    wav_path = '/export/fs05/lzhan268/workspace/PUBLIC/PartialSpoof/database/train/con_wav/CON_T_0000000.wav'  # noqa
    ori_sr, ori_wav = sciwav.read(wav_path)
    new_wav = process_Rawboost_feature(ori_wav, ori_sr, args, 5)


if __name__ == '__main__':
    debug()
