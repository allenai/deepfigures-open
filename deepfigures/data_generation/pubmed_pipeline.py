import collections
import datetime
import glob
import logging
import math
import multiprocessing
import os
import re
import subprocess
from typing import List, Tuple, Optional, Dict, Iterable

import bs4
from bs4 import BeautifulSoup
import cv2
import editdistance
import numpy as np
import scipy as sp
from PIL import Image
from botocore.vendored.requests.exceptions import ReadTimeout

from deepfigures import settings
from deepfigures.extraction import (
    figure_utils,
    exceptions,
    datamodels,
    renderers)
from deepfigures.utils import (
    stringmatch,
    config,
    traits,
    file_util,
    image_util,
    settings_utils)
from deepfigures.settings import (
    PUBMED_INPUT_DIR,
    PUBMED_INTERMEDIATE_DIR,
    PUBMED_DISTANT_DATA_DIR,
    LOCAL_PUBMED_DISTANT_DATA_DIR)


LOCAL_INTERMEDIATE_DIR = LOCAL_PUBMED_DISTANT_DATA_DIR + 'intermediate/'
LOCAL_FIGURE_JSON_DIR = LOCAL_PUBMED_DISTANT_DATA_DIR + 'figure-jsons/'

PDFTOTEXT_DPI = 72
MAX_PAGES = 50

pdf_renderer = settings_utils.import_setting(
    settings.DEEPFIGURES_PDF_RENDERER)()


def get_input_tars(suffix: str='') -> List[str]:
    """Returns a list of PMC source tarfiles, restricted to suffix (e.g. '00/')."""
    dirname = PUBMED_INPUT_DIR + suffix
    while True:
        try:
            return list(file_util.iterate_files(dirname))
        except ReadTimeout as e:
            logging.exception(
                'Timeout listing files in %s, retrying' % dirname, flush=True
            )


def get_result_jsons(prefix: str) -> List[str]:
    logging.info(datetime.datetime.now(), flush=True)
    search_dir = LOCAL_FIGURE_JSON_DIR + prefix
    jsons = sorted(glob.glob(search_dir))
    logging.info('Found %d jsons at %s' % (len(jsons), search_dir))
    logging.info(datetime.datetime.now(), flush=True)
    return jsons


def get_bin(pdf: str) -> str:
    """
    Get the bins of the pdf, e.g. './00/02/Br_J_Cancer_1977_Jan_35(1)_78-86.tar.gz'
    returns '00/02'.
    """
    parts = pdf.split('/')
    return parts[-3] + '/' + parts[-2] + '/'


class MatchedString(config.JsonSerializable):
    """A typed object representing the result of running stringmatch."""
    start_pos = traits.Int()
    end_pos = traits.Int()
    cost = traits.Int()

    @staticmethod
    def from_match(match) -> 'MatchedString':
        return MatchedString(
            start_pos=match.start_pos, end_pos=match.end_pos, cost=match.cost
        )


class PubmedMatchedFigure(config.JsonSerializable):
    """
    Contains data on a figure extracted from a PMC paper via caption matching with the included nxml file.
    """
    fig_im = traits.Instance(np.ndarray)
    page_image_name = traits.Unicode()
    caption = traits.Unicode()
    name = traits.Unicode()
    matched_caption = traits.Unicode()
    html_page = traits.Unicode()
    start_pos = traits.Int()
    end_pos = traits.Int()
    pdf = traits.Unicode()
    page_num = traits.Int()


def get_xml_soup(pdf: str) -> Optional[BeautifulSoup]:
    xml = pdf[:-4] + '.nxml'
    if not os.path.isfile(xml):
        return None
    with open(xml, 'r') as f:
        xml_soup = BeautifulSoup(f, 'xml')
    return xml_soup


def get_author_name(author: bs4.Tag) -> Optional[str]:
    """
    Given an xml tag representing an author, return that author's name as it will appear in the PDF, with any given
    names followed by surname.
    """
    surname = author.surname
    if surname is None:
        return None
    given_names = author.find_all('given-names')
    return ' '.join([name.text for name in given_names + [surname]])


def find_str_words_in_pdf(
    key: str,
    html_pages: List[bs4.Tag],
    pages: Optional[List[int]]=None,
    max_dist: int=math.inf,
) -> Tuple[Optional[List[bs4.Tag]], int]:
    if pages is None:
        pages = list(range(len(html_pages)))
    text_pages = [
        re.sub('\n', ' ', html_page.text) for html_page in html_pages
    ]
    clean_key = clean_str(key)
    matches = [
        MatchedString.
        from_match(stringmatch.match(clean_key, clean_str(page)))
        for (page_num, page) in enumerate(text_pages) if page_num in pages
    ]
    page_num = int(np.argmin([match.cost for match in matches]))
    match = matches[page_num]
    if match.cost > max_dist:
        matched_words = None
    else:
        matched_words = find_match_words(html_pages[page_num], match)
        matched_word_text = ' '.join([word.text for word in matched_words])
        if editdistance.eval(key, matched_word_text) > max_dist:
            matched_words = None
    return matched_words, page_num


def find_match_words(page: bs4.Tag, match: MatchedString) -> List[bs4.Tag]:
    words = page.find_all('word')
    start_pos = 0
    start_token_idx = 0
    while start_pos < match.start_pos:
        # Start at the end of partially matching tokens
        start_pos += len(clean_str(words[start_token_idx].text))
        start_token_idx += 1
    end_pos = start_pos
    end_token_idx = start_token_idx
    while end_pos < match.end_pos:
        # Stop at the end of partially matching tokens
        end_pos += len(clean_str(words[end_token_idx].text))
        end_token_idx += 1
    matching_words = words[start_token_idx:end_token_idx]
    return matching_words


def words_to_box(
    words: Optional[List[bs4.Tag]], target_dpi=settings.DEFAULT_INFERENCE_DPI
) -> Optional[datamodels.BoxClass]:
    if words is None or len(words) == 0:
        return None
    word_boxes = [
        datamodels.BoxClass.from_xml(word, target_dpi) for word in words
    ]
    return datamodels.enclosing_box(word_boxes)


def tag_to_tokens(tag: bs4.Tag) -> Iterable[str]:
    for c in tag.contents:
        if type(c) == bs4.NavigableString:
            s = c
        elif hasattr(c, 'text'):
            s = c.text
        else:
            s = ''
        for token in s.split():
            yield token


def match_figures(pdf: str, ignore_errors=False
                 ) -> Optional[Dict[str, List[datamodels.Figure]]]:
    print('.', end='', flush=True)
    logging.info('Matching figures for %s' % pdf)
    try:
        page_names = pdf_renderer.render(
            pdf_path=pdf,
            output_dir=os.path.dirname(pdf),
            dpi=settings.DEFAULT_INFERENCE_DPI,
            max_pages=MAX_PAGES,
            check_retcode=True
        )
    except subprocess.CalledProcessError:
        logging.exception('Failed to render pdf: %s' % pdf)
        return None
    try:
        xml_soup = get_xml_soup(pdf)
        if xml_soup is None:
            # This can be caused by files with multiple PDFs
            logging.info('No xml soup found for %s' % pdf)
            return None
        html_soup = pdf_renderer.extract_text(pdf)
        if html_soup is None:
            # pdftotext fails on some corrupt pdfs
            logging.warning('Pdftotext failed, pdf corrupt: %s' % pdf)
        html_pages = html_soup.findAll('page')
        xml_figures = xml_soup.findAll('fig')
        xml_tables = xml_soup.findAll('table-wrap')
        matched_figures = []
        for xml_fig in xml_figures + xml_tables:
            matched_figure = match_figure(xml_fig, html_pages, pdf, page_names)
            if matched_figure is None:
                return None
            else:
                matched_figures.append(matched_figure)
        if len(matched_figures) == 0:
            # Some papers contain figures but don't use standard XML tags
            return None
        else:
            figures_by_page = {page_name: []
                               for page_name in page_names
                              }  # type: Dict[str, List[datamodels.Figure]]
            for fig in matched_figures:
                figures_by_page[page_names[fig.page]].append(fig)
            return figures_by_page
    except Exception:
        logging.exception('Exception for pdf %s' % pdf)
        if ignore_errors:
            return None
        else:
            raise


def match_figure(
    xml_fig: bs4.Tag,
    html_pages: List[bs4.Tag],
    pdf: str,
    page_names: Optional[List[str]]=None
) -> Optional[datamodels.Figure]:
    if xml_fig.caption is None or xml_fig.label is None:
        # Some tables contain no caption
        logging.warning(
            'No caption or label found for %s in %s' % (xml_fig.name, pdf)
        )
        return None
    label = xml_fig.label.text
    caption = label + ' ' + xml_fig.caption.text
    caption_words, page_num = find_str_words_in_pdf(caption, html_pages)
    caption_boundary = words_to_box(caption_words)
    if caption_boundary is None:
        logging.warning('Failed to locate caption for %s in %s' % (label, pdf))
        return None
    html_page = html_pages[page_num]
    page_words = html_page.find_all('word')
    words_inside_box = [
        word for word in page_words
        if caption_boundary.contains_box(datamodels.BoxClass.from_xml(word))
    ]
    if len(words_inside_box) / len(caption_words) > 1.5:
        logging.warning(
            '%s in %s includes too many non-caption words: %f' %
            (label, pdf, len(words_inside_box) / len(caption_words))
        )
    if page_num >= MAX_PAGES:  # page_num is 0 indexed
        return None
    page_im = image_util.read_tensor(page_names[page_num])
    page_height, page_width = page_im.shape[:2]
    if xml_fig.graphic is not None:
        image_name = xml_fig.graphic.get('xlink:href')
        if image_name is None:
            image_name = xml_fig.graphic.get('href')
        if image_name is None:
            logging.warning('Figure graphic contains no image')
            return None
        fig_image_name = os.path.dirname(pdf) + '/' + image_name + '.jpg'
        if not os.path.isfile(fig_image_name):
            logging.warning('Image file not found for %s in %s' % (label, pdf))
            return None
        fig_im = image_util.read_tensor(fig_image_name)
        figure_boundary = find_fig_box(fig_im, page_im)
        if figure_boundary is None:
            logging.warning(
                'Failed to match figure for %s in %s' % (label, pdf)
            )
            return None
    elif xml_fig.name == 'table-wrap':
        # Need to search for footer and table separately since they can be separated in the token stream
        table_header = xml_fig.find_all('th')
        table_body = xml_fig.find_all('td')
        table_footer = xml_fig.find_all('table-wrap-foot')
        table_tokens = [
            token
            for t in table_header + table_body for token in tag_to_tokens(t)
        ]
        footer_tokens = [
            token for t in table_footer for token in t.text.split()
        ]
        page_table_content_words, content_dist = find_page_table_words(
            table_tokens, page_words
        )
        page_table_footer_words, footer_dist = find_page_table_words(
            footer_tokens, page_words
        )
        total_dist = content_dist + footer_dist
        total_tokens = len(table_tokens) + len(footer_tokens)
        if total_tokens == 0:
            logging.warning(
                'Failed to match any table contents for %s in %s' %
                (label, pdf)
            )
            return None
        if total_dist / total_tokens > .5:
            logging.warning(
                '%s in %s table is too far from the xml table: %f' %
                (label, pdf, total_dist / total_tokens)
            )
            return None
        page_table_words = page_table_content_words + page_table_footer_words
        figure_boundary = words_to_box(page_table_words)
        words_inside_box = [
            word for word in page_words
            if
            figure_boundary.contains_box(datamodels.BoxClass.from_xml(word))
        ]
        if len(words_inside_box) / total_tokens > 1.2:
            logging.warning(
                '%s in %s includes too many non-table words: %f' %
                (label, pdf, len(words_inside_box) / total_tokens)
            )
            return None
    else:
        logging.warning('No graphic found for %s' % pdf)
        return None

    if xml_fig.name == 'fig':
        figure_type = 'Figure'
    else:
        assert xml_fig.name == 'table-wrap'
        figure_type = 'Table'
    return datamodels.Figure(
        figure_boundary=figure_boundary,
        caption_boundary=caption_boundary,
        caption_text=caption,  # Caption from the PubMed XML
        name=label,
        page=page_num,
        figure_type=figure_type,
        dpi=settings.DEFAULT_INFERENCE_DPI,
        page_height=page_height,
        page_width=page_width
    )


def find_page_table_words(table_tokens: List[str], page_words: List[bs4.Tag]) -> \
        Tuple[List[bs4.Tag], int]:
    if len(table_tokens) == 0:
        return [], 0
    assert len(page_words) > 0
    table_token_counter = collections.Counter(table_tokens)
    table_token_count = sum(table_token_counter.values())
    page_tokens = [word.text for word in page_words]

    best_dist = math.inf
    best_seq = None
    for start_idx in range(len(page_words)):
        diff_counter = table_token_counter.copy()
        cur_dist = table_token_count
        for end_idx in range(start_idx + 1, len(page_tokens)):
            cur_token = page_tokens[end_idx - 1]
            token_count = diff_counter[cur_token]
            diff_counter[cur_token] = token_count - 1
            if token_count <= 0:
                cur_dist += 1
            else:
                cur_dist -= 1
            if cur_dist < best_dist:
                best_dist = cur_dist
                best_seq = (start_idx, end_idx)
    assert best_seq is not None
    best_start, best_end = best_seq
    return page_words[best_start:best_end], best_dist


def clean_str(s: str) -> str:
    # Some figures have labels with mismatching cases
    # so we should be case insensitive
    return ''.join(s.split()).lower()


SCALE_FACTOR = .1


def find_template_in_image(fig_im: np.ndarray, page_im: np.ndarray, scales: List[float], use_canny: bool) -> \
        Optional[Tuple[datamodels.BoxClass, float, float]]:
    """
    Find the position of the best match for fig_im on page_im by checking at each of a list of scales.
    Each scale is a float in (0,1] representing the ratio of the size of fig_im to page_im (maximum of height ratio and
    width ratio).
    """
    try:
        template = sp.misc.imresize(fig_im, SCALE_FACTOR)
    except ValueError:
        # This may cause some very small images to have size 0 which causes a ValueError
        return None
    (template_height, template_width) = template.shape[:2]
    (page_height, page_width) = page_im.shape[:2]
    if use_canny:
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        template = cv2.Canny(template, 100, 200)
        page_im = cv2.cvtColor(page_im, cv2.COLOR_BGR2GRAY)
    found = None
    best_scale = None
    template_page_size_ratio = max(
        template_height / page_height, template_width / page_width
    )
    # loop over the scales of the image
    for scale in (scales)[::-1]:
        # resize the image according to the scale, and keep track of the ratio of the resizing.
        page_resized = sp.misc.imresize(
            page_im, template_page_size_ratio / scale
        )
        r = page_im.shape[1] / float(page_resized.shape[1])
        assert (
            page_resized.shape[0] >= template_height and
            page_resized.shape[1] >= template_width
        )
        if use_canny:
            page_resized = cv2.Canny(page_resized, 50, 200)
        result = cv2.matchTemplate(
            page_resized, template, cv2.TM_CCOEFF_NORMED
        )
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)
            best_scale = scale
            logging.debug('Scale: %.03f, Score: %.03f' % (scale, maxVal))

    assert found is not None
    (score, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (
        int((maxLoc[0] + template_width) * r),
        int((maxLoc[1] + template_height) * r)
    )
    fig_box = datamodels.BoxClass(x1=startX, y1=startY, x2=endX, y2=endY)
    return fig_box, score, best_scale


def find_fig_box(
    fig_im: np.ndarray, page_im: np.ndarray, use_canny: bool=False
) -> Optional[datamodels.BoxClass]:
    """Find the position of the best match for fig_im on page_im through multi scale template matching."""
    # If we get a score below this threshold, it's probably a bad detection
    score_threshold = 0.8
    scales = np.concatenate(
        (
            np.logspace(np.log10(.1), np.log10(.2), 5),
            np.logspace(np.log10(.2), np.log10(.95), 40)
        ),
        axis=0
    )  # type: List
    res = find_template_in_image(fig_im, page_im, scales, use_canny)
    if res is None:
        return None
    fig_box, score, best_scale = res
    refined_scales = [
        scale
        for scale in np.linspace(.97 * best_scale, 1.03 * best_scale, 16)
        if scale <= 1.0
    ]
    (refined_fig_box, refined_score,
     best_refined_scale) = find_template_in_image(
         fig_im, page_im, refined_scales, use_canny
     )
    if refined_score < score_threshold:
        return None
    else:
        return refined_fig_box


def run_full_pipeline(
    tarpath: str, skip_done: bool=True, save_intermediate: bool=False
) -> None:
    foldername = str(os.path.basename(tarpath).split('.')[0])
    result_path = LOCAL_FIGURE_JSON_DIR + get_bin(
        tarpath
    ) + foldername + '.json'
    if skip_done and file_util.exists(result_path):
        return
    d = LOCAL_INTERMEDIATE_DIR + get_bin(tarpath)
    while True:
        try:
            file_util.extract_tarfile(tarpath, d, streaming=False)
            # botocore.vendored.requests.packages.urllib3.exceptions.ReadTimeoutError can't be caught because it doesn't
            # inherit from BaseException, so don't use streaming
            break
        except FileNotFoundError as e:
            logging.exception('Failure reading %s, retrying' % tarpath)
        except ReadTimeout as e:
            logging.exception('Timeout reading %s, retrying' % tarpath)
    pdfs = glob.glob(d + foldername + '/' + '*.pdf')
    res = dict()
    for pdf in pdfs:
        sha1sum = file_util.compute_sha1(pdf)
        with open(pdf + '.sha1', 'w') as f:
            print(sha1sum, file=f)
        paper_figures = match_figures(pdf)
        if paper_figures is not None:
            res.update(paper_figures)
    if save_intermediate:
        intermediate_path = PUBMED_INTERMEDIATE_DIR + get_bin(
            tarpath
        ) + foldername + '/'
        for file in glob.glob(d + '/' + foldername + '/' + '*'):
            file_util.copy(file, intermediate_path + os.path.basename(file))
    file_util.safe_makedirs(os.path.dirname(result_path))
    file_util.write_json_atomic(
        result_path,
        config.JsonSerializable.serialize(res),
        indent=2,
        sort_keys=True
    )


def run_on_all() -> None:
    Image.MAX_IMAGE_PIXELS = int(1e8)  # Don't render very large PDFs.
    Image.warnings.simplefilter('error', Image.DecompressionBombWarning)

    print(datetime.datetime.now())
    print('Starting', flush=True)
    topdirs = ['%.2x' % n for n in range(256)]
    dirs_per_partition = 32
    for partition in range(0, len(topdirs), dirs_per_partition):
        curdirs = topdirs[partition:partition + dirs_per_partition]
        print(datetime.datetime.now())
        print('Processing dirs: %s' % str(curdirs))
        with multiprocessing.Pool() as p:
            nested_tarfiles = p.map(
                get_input_tars, [topdir for topdir in curdirs]
            )
        tarfiles = [t for tarfiles in nested_tarfiles for t in tarfiles]
        assert len(tarfiles) == len(set(tarfiles))
        print(datetime.datetime.now())
        print('Processing %d tarfiles in %s' % (len(tarfiles), str(curdirs)))
        with multiprocessing.Pool(processes=round(1.5 * os.cpu_count())) as p:
            p.map(run_full_pipeline, tarfiles)
        print('All done')


if __name__ == "__main__":
    logging.basicConfig(filename='logger_pubmed.log', level=logging.WARNING)
    run_on_all()
