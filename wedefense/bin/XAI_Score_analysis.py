import argparse
import pickle
import os
import math
def time_to_frame(time, shift=0, max_length=99999):
    # mapping, 10 -> consider the window. wav2vec2 -> 20ms/frame
    return min(max(int((time * 1000 - 10) // 20 + shift), 0), max_length)

def main(set_name, pkl_path, vad_path):
    with open(pkl_path, 'rb') as f:
        items = pickle.load(f)

    # Index: 0real, 1fake, 2mix, 3slireal, 4slifake
    index_type = ['Bona fide Speech (BS)    ', 
                  'Spoofed Speech (SS)      ', 
                  'Tansition Region (TR)    ',
                  'Bona fide Non-speech (BN)',
                  'Spoofed Non-speech (SN)  ']
    n_times_list = [0] * 5
    n_frames_list = [0] * 5
    values_list = [0.0] * 5

    for item in items:
        filename = item[0][0]
        data = item[1]
        n_frames = len(data) - 2
        
        if filename[:3] == 'CON':  # partial spoof samples only
            with open(os.path.join(vad_path, set_name, filename + '.vad'), "r") as f_vad:
                vads = f_vad.readlines()

            # To avoid #frame mismatches with VAD labels.
            parts = vads[-1].strip().split(' ')
            end_time = float(parts[1])    
            num_frames = math.ceil(float(end_time)*1000/20)
            if len(data)-num_frames>0:
                data = data[len(data)-num_frames:]

            # follow the interspeech paper, this is for the data to draw the figure.
            if filename == "CON_E_0033629": 
                print("For the visualization of CON_E_0033629: GradCam Scores:", data, 'labels:', vads)

            for line in vads:
                parts = line.strip().split(' ')
                st_time = float(parts[0])
                end_time = float(parts[1])
                label = int(parts[-1])
                
                st_frame = time_to_frame(st_time, shift=0, max_length=n_frames)
                end_frame = time_to_frame(end_time, shift=0, max_length=n_frames)

                if label == 1:
                    n_times_list[0] += 1
                    n_frames_list[0] += max(end_frame - st_frame - 1, 0)
                    for i in range(st_frame + 1, end_frame):
                        values_list[0] += float(data[i])

                elif 2 <= label <= 20:
                    n_times_list[1] += 1
                    n_frames_list[1] += max(end_frame - st_frame - 1, 0)
                    for i in range(st_frame + 1, end_frame):
                        values_list[1] += float(data[i])

                elif label == 100:
                    n_times_list[2] += 1
                    n_frames_list[2] += max(end_frame - st_frame - 1, 0)
                    for i in range(st_frame + 1, end_frame):
                        values_list[2] += float(data[i])

                elif label == 101:
                    n_times_list[3] += 1
                    n_frames_list[3] += max(end_frame - st_frame + 1, 0)
                    for i in range(st_frame, end_frame + 1):
                        values_list[3] += float(data[i])

                elif 102 <= label <= 120:
                    n_times_list[4] += 1
                    n_frames_list[4] += max(end_frame - st_frame + 1, 0)
                    for i in range(st_frame, end_frame + 1):
                        values_list[4] += float(data[i])

                else:
                    print(f"Unknown label: {label}")

    f = [(v / n if n > 0 else 0.0) for v, n in zip(values_list, n_frames_list)]

    avg = sum(values_list[0:5]) / sum(n_frames_list[0:5])
    print('--------------------{}---------------------'.format(set_name))
    print('Relative Contribution Quantification (RCQ):')
    for i in range(len(f)):
        print('{}: {:.2f}%'.format(index_type[i], (f[i] - avg) / avg * 100))
    print('-----------------------------------------')

    # print('avg: {:.4f}'.format(avg))

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', required=True, help='eval or dev')
    parser.add_argument('--pkl_path', required=True, help='Path to the grad-CAM score .pkl file')
    parser.add_argument('--vad_path', required=True, help='Directory containing .vad files')
    args = parser.parse_args()

    main(args.set, args.pkl_path, args.vad_path)