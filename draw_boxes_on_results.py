import glob, os, json, sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_bounding_box(image_path, x1, y1, x2, y2):
    im = np.array(Image.open(image_path), dtype=np.uint8)
    fig, ax = plt.subplots(1)
    ax.imshow(im)
    width = x2 - x1
    height = y2 - y1
    rect = patches.Rectangle((x1, y1), width, height, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show()


def obtain_figure_paths(result_dir):
    pdf_path = glob.glob(os.path.join(result_dir, '*.pdf'))[0]
    pdf_name = os.path.split(pdf_path)[1].split('.pdf')[0]
    figure_dir = os.path.join(result_dir, pdf_name + '.pdf-images', 'ghostscript', 'dpi100')
    figure_paths_ = glob.glob(os.path.join(figure_dir, '*.png'))
    figure_paths_.sort()
    return figure_paths_


def obtain_result_json(result_dir):
    json_path = glob.glob(os.path.join(result_dir, '*.json'))[0]
    return json.load(open(json_path))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: 1 argument needed. i.e. results dir path.")
        print("Correct invocation:")
        print(" ".join(["python", __file__, "<path_to_results_directory>"]))
        exit(1)

    result_folder = os.path.abspath(str(sys.argv[1]))

    figure_paths = obtain_figure_paths(result_dir=result_folder)
    result_json = obtain_result_json(result_dir=result_folder)
    assert len(result_json['raw_detected_boxes']) == len(figure_paths), "Num pages not equal to num results."

    for figure_path in figure_paths:
        fig_idx = int(figure_path.split('-')[-1].split('.png')[0].split('page')[1]) - 1
        page_bbs = result_json['raw_detected_boxes'][fig_idx]
        for bb in page_bbs:
            plot_bounding_box(figure_path, x1=bb['x1'], y1=bb['y1'], x2=bb['x2'], y2=bb['y2'])
