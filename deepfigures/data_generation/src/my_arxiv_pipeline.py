import logging, os
from PIL import Image
from deepfigures.utils import file_util
from deepfigures.extraction import figure_utils
import multiprocessing
import multiprocessing.pool
import subprocess
import json

ARXIV_TAR_SRC = 's3://arxiv/src/'
DOWNLOAD_FOLDER = ''


def _parse_s3_location(path):
    logging.debug('Parsing path %s' % path)
    if not path.startswith('s3://'):
        raise ValueError('s3 location must start with s3://')

    path = path[5:]
    parts = path.split('/', 1)
    if len(parts) == 1:
        bucket = parts[0]
        key = None
    else:
        bucket, key = parts

    return {'bucket': bucket, 'key': key}


def download_tar(tar_name):
    print("Downloading", tar_name)

    parse = _parse_s3_location(tar_name)
    target_filename = parse['key'].split('/')[1]

    command = 'aws2 s3api get-object --bucket "%s" --key "%s" "%s" --request-payer=requester' % (
        parse['bucket'], parse['key'], target_filename)

    retcode = subprocess.call(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        shell=True
    )

    if retcode == 0:
        print("Successfully downloaded {}. Nov doing scp.".format(parse['key']))
        scp_command = 'scp "%s" cascades2.arc.vt.edu:/home/sampanna/arxiv_data' % target_filename
        scp_retcode = subprocess.call(
            scp_command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=True
        )
        if scp_retcode == 0:
            print("scp successful for {}".format(target_filename))
            rm_command = 'rm "%s"' % target_filename
            rm_retcode = subprocess.call(
                rm_command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=True
            )
            if rm_retcode == 0:
                print("Successfully removed {}".format(target_filename))
            else:
                print("Failed to remove: {}".format(target_filename))
        else:
            print("Non-zero return code for scp. Filename: {}".format(target_filename))

    if retcode != 0:
        print("Failed to download: {}".format(parse['key']))

    return 'Finished'


def execute_parallel(function_reference, args, threads_per_cpu=2):
    with multiprocessing.Pool(processes=round(threads_per_cpu * os.cpu_count())) as pool:
        pool.map(function_reference, args)


if __name__ == "__main__":
    logging.basicConfig(filename='logger_arxiv.log', level=logging.WARNING)
    Image.MAX_IMAGE_PIXELS = int(1e8)  # Don't render very large PDFs.

    Image.warnings.simplefilter('error', Image.DecompressionBombWarning)

    tar_names = []
    with open('tar_names.json', mode='r') as tar_names_json:
        tar_names = tar_names + json.loads(tar_names_json.read())

    execute_parallel(download_tar, tar_names)
