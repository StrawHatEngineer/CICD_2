from instabase.ocr.client.libs.ibocr import ParsedIBOCRBuilder 
from instabase.provenance.registration import register_fn
from instabase.udf_utils.clients.udf_helpers import get_output_ibmsg 
import logging 
import json 

@register_fn(provenance=False)
def write_summary(**kwargs): 
  logging.info('Started running Reduce UDF function') 
  summary_dict = {} 
  fn_context = kwargs.get('_FN_CONTEXT_KEY') 
  input_records, _ = fn_context.get_by_col_name('INPUT_RECORDS') 
  for record in input_records: 
    input_filepath = record['input_filepath'] 
    output_filename = record['output_filename'] 

    # Loading ibmsg and then we can get records from it. 
    builder, err = ParsedIBOCRBuilder.load_from_str(input_filepath, record['content']) 
    for ibocr_record in builder.get_ibocr_records(): 
      raw_input_filepath = ibocr_record.get_document_path() 
      raw_file_name = raw_input_filepath.split('/')[-1]
      if raw_input_filepath not in summary_dict: 
        summary_dict[raw_file_name] = []  
      result = {} 

      if ibocr_record.has_class_label():  
        result['class_label'] = ibocr_record.get_class_label() 
        if ibocr_record.has_classify_page_range(): 
          result['page_range'] = ibocr_record.get_classify_page_range() 
        else: 
          result['page_range'] = {} 
          result['page_range']['start_page'] = ibocr_record.get_page_numbers()[0]+1 
          result['page_range']['end_page'] = ibocr_record.get_page_numbers()[-1]+1 
      result['extracted_fields'] = {} 

      refined_phrases, _ = ibocr_record.get_refined_phrases() 
      for phrase in refined_phrases: 
        name = phrase.get_column_name()
        if name.startswith('__'):
          continue
        value = phrase.get_column_value() 
        result['extracted_fields'][name] = value 
      summary_dict[raw_file_name].append(result) 

      # we don't want to modify the input ibmsg here, just want to let it flow through to the next step  
      # use yield here to avoid OOM when processing a large amount of files 
      step_out_folder, _ = fn_context.get_by_col_name('STEP_FOLDER') 
      output_ibmsg, err = get_output_ibmsg(record['input_filepath'], step_out_folder, record['content']) 

      yield { 
        'out_files': [{ 
            'filename': output_filename, 
            'content': output_ibmsg 
        }] 
      }

  clients, _ = fn_context.get_by_col_name('CLIENTS') 
  root_out, _ = fn_context.get_by_col_name('ROOT_OUTPUT_FOLDER') 

  # we don't want the summary file to be passed to the next step of the flow, so write it out using ibfile 
  clients.ibfile.write_file(root_out.strip('/')+f'/summary/summary.json', json.dumps(summary_dict, indent=4))  

  return  
