from typing import Any
from instabase.provenance.registration import register_fn
import logging
from instabase.data_api_utils._sheet import confidence_fns
from instabase.ocr.client.libs.algorithms import WordPolyInputColMapper
import json


@register_fn(provenance=False)
def print_name(text, *args: Any, **kwargs: Any) -> None:
  logging.info("Hi my name is vinay!")
  return text

@register_fn
def get_text_from_coordinates(bbox, page_num=0, th=10, **kwargs):
    logger, err = kwargs['_FN_CONTEXT_KEY'].get_by_col_name('LOGGER')
    input_ibocr, err = kwargs['_FN_CONTEXT_KEY'].get_by_col_name('INPUT_IBOCR')
    # bbox = bbox.value()
    bbox = dict(bbox)
    extracted_metadata = []
    extracted_metadata_by_line = []
    word_list = []
    all_keys = input_ibocr.keys()
    # logging.info(list(all_keys))
    # return(input_ibocr)
    # return(input_ibocr['lines'])
    ibocr_lines = input_ibocr['lines']
    data_control_code_found = {'Data':False,'Control':False,'Codes':False}
    for line_num, line in enumerate(ibocr_lines):
      for word_num, word in enumerate(line):
        if word['page'] != page_num:
          continue
        logging.info(word)
        logging.info(bbox)
        if (word['start_x'] > bbox['x']-th) and (word['end_x'] < bbox['end_x']+th) and (word['start_y'] > bbox['y']-th) and (word['end_y'] < bbox['end_y']+th):
          word_list.append(word)

    ibocr_record, err = kwargs['_FN_CONTEXT_KEY'].get_by_col_name('INPUT_IBOCR_RECORD')
    mapper = WordPolyInputColMapper(ibocr_record)
    # logging.info(word_list)
    return mapper.get_cluster([word_list])

@register_fn
def return_extracted_positions(s, **kwargs):
  ibocr_record, err = kwargs['_FN_CONTEXT_KEY'].get_by_col_name('INPUT_IBOCR_RECORD')
  record = ibocr_record
  retval = confidence_fns.get_extracted_region(record, s.tracker().get_iterator_over_extracted_with_current())
  if retval[1]:
    create_iberror_from_refiner(retval[1])
  return retval[0]

@register_fn
def compute_text_region(**word_details):

    logging.info(word_details)

    # Extract relevant details
    start_x = word_details['start_x']
    start_y = word_details['start_y']
    end_x = word_details['end_x']
    end_y = word_details['end_y']
    char_width = word_details['char_width']
    word_width = word_details['word_width']
    line_height = word_details['line_height']

    # Compute column, row, width, and height
    column = int(start_x // char_width)
    row = int(start_y // line_height)
    width = int(word_width // char_width)
    height = int((end_y - start_y) // line_height) or 1

    return [column, row, width, height]
