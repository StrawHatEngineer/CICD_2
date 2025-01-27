from ib.market.ib_image_signature_comparator.register import register_all
from instabase.provenance.registration import register_fn
import os 
import logging

def register(name_to_fn):
    register_all(name_to_fn)

@register_fn(provenance=False)
def get_user(**kwargs) -> str:
  fn_context = kwargs.get("_FN_CONTEXT_KEY")
  input_filepath, _ = fn_context.get_by_col_name('INPUT_FILEPATH')
  base_filename = os.path.basename(input_filepath)
  doc_name = base_filename.split('_')[0]

  root_folder = os.path.dirname(input_filepath)
  path = os.path.join(root_folder, doc_name)
  return path

@register_fn(provenance=False)
def verify_signs(similarity_scores, threshold, **kwargs) -> str:
  logging.info(similarity_scores)
  if max(similarity_scores) > threshold:
    return "Genuine"
  else:
    return "Forged"