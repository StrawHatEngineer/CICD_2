from instabase.udf_utils.clients.udf_helpers import get_output_ibmsg

def map_udf_func(input_record, step_folder, *args, **kwargs):
  input_filepath = input_record['input_filepath']
  file_content = input_record['content']
  output_filename = input_record['output_filename']

  # Copy output ibmsg from input
  output_ibmsg, err = get_output_ibmsg(input_filepath, step_folder, file_content)

  return {
    "out_files": [
      {
        "filename": output_filename,
        "content": output_ibmsg
      }
    ]
  }

def register(name_to_fn):
  name_to_fn.update({
    'map_udf': {
      'fn': map_udf_func
    }
  })