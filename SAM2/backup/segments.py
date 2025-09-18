import os
import sys

def extract_phases(txt_file_path):
    phases_dict = {}

    with open(txt_file_path, 'r') as file:
        lines = file.readlines()

    for line in lines[1:]:
        frame_index, phase = line.strip().split()
        frame_index = int(frame_index)

        if frame_index % 25 == 0:
            subsample_index = frame_index // 25
            if phase not in phases_dict:
                phases_dict[phase] = []
            phases_dict[phase].append(subsample_index)

    return phases_dict

def segment_by_phases(folder_path, txt_file_path):
    # Check if the folder and txt file correspond to the same video
    folder_video = os.path.basename(folder_path)
    txt_video = os.path.basename(txt_file_path).split('-')[0]
    
    if folder_video != txt_video:
        raise ValueError("The folder and text file do not correspond to the same video.")

    # Extract phases, key will be the phase and value will be a list of indices
    phases_dict = extract_phases(txt_file_path)

    # Store phase-images segments
    segments_dict = {phase: [[]] for phase in phases_dict}

    png_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])

    # Map the phase dict to the png files
    for phase, indices in phases_dict.items():
        for index in indices:
            png_file = f"{index:05d}.png"
            if png_file in png_files:
                if len(segments_dict[phase][-1]) >= 500:
                    segments_dict[phase].append([])
                segments_dict[phase][-1].append(png_file)

    return segments_dict

def main():
    import sys
    if len(sys.argv) != 3:
        print("Usage: python segments.py <folder_path> <txt_file_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    txt_file_path = sys.argv[2]

    segments_dict = segment_by_phases(folder_path, txt_file_path)
    for key, value in segments_dict.items():
        print(key, len(value), value[-1], '\n')


if __name__ == "__main__":
    main()