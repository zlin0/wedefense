import argparse
import pickle
import os

def time_to_frame(time, shift=0, max_length=99999):
    return min(max(int((time * 1000 - 5) // 20 + shift), 0), max_length)

def main(pkl_path, vad_path, class_for_grad=2):
    with open(pkl_path, 'rb') as f:
        items = pickle.load(f)

    n_times_list = [0] * 7
    n_frames_list = [0] * 7
    values_list = [0.0] * 7

    for item in items:
        n_frames = len(item) - 2
        item = item[1:]
        filename = item[0][0]

        if filename[:3] != 'CON':  # Only fake samples (you can reverse this)
            continue

        with open(os.path.join(vad_path, filename + '.vad'), 'r') as f:
            vads = f.readlines()

        values_list[5] += float(item[0][class_for_grad])  # start
        values_list[6] += float(item[-1][class_for_grad])  # end
        n_frames_list[5] += 1
        n_frames_list[6] += 1

        for line in vads:
            parts = line.strip().split()
            start_time, end_time, label = float(parts[0]), float(parts[1]), int(parts[2])
            st_frame = time_to_frame(start_time, max_length=n_frames)
            end_frame = time_to_frame(end_time, max_length=n_frames)

            label_to_index = {
                1: 0,                     # fake
                range(2, 21): 1,          # spoof
                100: 2,                  # mix
                101: 3,                  # slireal
                range(102, 121): 4       # slifake
            }

            for label_range, idx in label_to_index.items():
                if isinstance(label_range, range):
                    if label in label_range:
                        break
                elif label == label_range:
                    break
            else:
                print(f"Unknown label: {label}")
                continue

            n_times_list[idx] += 1
            frame_count = max(end_frame - st_frame - (0 if idx == 2 else 1), 0)
            n_frames_list[idx] += frame_count
            for i in range(st_frame + (0 if idx == 2 else 1), end_frame + (1 if idx == 2 else 0)):
                values_list[idx] += float(item[i][class_for_grad])

    avg_values = [v / f if f > 0 else 0.0 for v, f in zip(values_list, n_frames_list)]
    for i, val in enumerate(avg_values):
        print(f"Type {i}: {val:.4f}")

    overall_avg = sum(values_list[:5]) / sum(n_frames_list[:5]) if sum(n_frames_list[:5]) > 0 else 0.0
    print(f"Overall avg: {overall_avg:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_path', required=True, help='Path to the grad-CAM score .pkl file')
    parser.add_argument('--vad_path', required=True, help='Directory containing .vad files')
    parser.add_argument('--class_for_grad', type=int, default=2, help='Column index of grad score in the item')
    args = parser.parse_args()

    main(args.pkl_path, args.vad_path, args.class_for_grad)