from pathlib import Path
import json
import os

# import tensorboxresnet.utils.annolist.AnnotationLib as al

figure_json_path = '/work/host-input/arxiv_data_output/figure-jsons'
output_figure_boundaries_path = '/work/host-input/figure_boundaries.json'
output_caption_boundaries_path = '/work/host-input/caption_boundaries.json'

figure_boundaries = []
caption_boundaries = []

for filename in Path(figure_json_path).rglob('*.json'):
    contents = json.load(open(str(filename)))
    for key, value in contents.items():
        dir_name, file = os.path.split(key)
        correct_path = os.path.join(dir_name, 'black.pdf-images/ghostscript', file)

        if not len(value):
            continue
        figure_annotation = {
            "image_path": correct_path,
            "rects": [ann['figure_boundary'] for ann in value]
        }
        caption_annotation = {
            "image_path": correct_path,
            "rects": [ann['caption_boundary'] for ann in value]
        }
        figure_boundaries.append(figure_annotation)
        caption_boundaries.append(caption_annotation)

json.dump(figure_boundaries, open(output_figure_boundaries_path, mode='w'))
json.dump(caption_boundaries, open(output_caption_boundaries_path, mode='w'))

# annolist = al.parse(output_figure_boundaries_path, abs_path=False)