from pathlib import Path
import json

# import tensorboxresnet.utils.annolist.AnnotationLib as al

path = '/home/sampanna/deepfigures-results/arxiv_data_output/figure-jsons'

figure_boundaries = []
caption_boundaries = []

for filename in Path(path).rglob('*.json'):
    contents = json.load(open(str(filename)))
    for key, value in contents.items():
        if not len(value):
            continue
        figure_annotation = {
            "image_path": key,
            "rects": [ann['figure_boundary'] for ann in value]
        }
        caption_annotation = {
            "image_path": key,
            "rects": [ann['caption_boundary'] for ann in value]
        }
        figure_boundaries.append(figure_annotation)
        caption_boundaries.append(caption_annotation)
# print(json.dumps(figure_boundaries))

# json.dump(figure_boundaries, open('delete_this.json', mode='w'))
#
# annolist = al.parse('/home/sampanna/workspace/bdts2/deepfigures-second/deepfigures-open/delete_this.json', abs_path=True)
