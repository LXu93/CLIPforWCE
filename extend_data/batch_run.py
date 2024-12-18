import os



root_dir = 'Frames_and_videos'
filetype = '.jpg'
def get_names(root_dir,filetype):  
    # Walk through the directory
    file_names = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(filetype):
                # Get full path and append it to the list
                full_path = os.path.join(dirpath, filename)
                file_names.append(full_path[:-4])
    
    return file_names

all_files = get_names(root_dir, filetype)

for path in all_files:
    split_text = path.split('\\')
    name = split_text[-3] + '_' +split_text[-1]
    os.system(f'python ci2v_lpips.py --image {path}.jpg --video {path}.mpg --number 10 --output extended_images\\' + name+ '.jpg')