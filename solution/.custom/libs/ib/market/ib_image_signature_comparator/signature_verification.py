import base64
import io
import json
from PIL import Image
import base64
import logging

from typing import Callable, List, Union

from google.protobuf import json_format
from instabase.provenance.tracking import Value, ImageProvenanceTracker

from .signature_utils import _create_image_position, _get_pil_image_from_ibocr, sign_clean_impl, sign_verify, find_threshold, find_similarity, rotate_image

to_json: Callable = lambda proto: json.loads(
  json_format.MessageToJson(proto, preserving_proto_field_name=True))

# --------------------------------------------------------
# --------- Signature Extraction -------------------------
# --------------------------------------------------------

def extract_signature_crop(field_value: Value[str], **kwargs) -> Value[str]:
  
  """
  Gets an image patch of the input text as a base64-encoded string.
  ```
  Args:
      field_value (Value[Text]): A provenance-tracked text to extract a patch of signature  

  Returns:
      Value[str]: Returns a base64-encoded string of the extracted image patch

  Examples:
      extract_signature_crop(list_get(match(INPUT_COL, '[SIGN]'), 0))
      -> '/9j/4AAQSkZ...'
  ```
  """

  fn_context = kwargs.get("_FN_CONTEXT_KEY")
  refiner, error = fn_context.get_by_col_name("REFINER_FNS")
  ibfile = fn_context.get_ibfile()
  orig_ibocr, err = fn_context.get_by_col_name('ORIG_IBOCR')
  record, err = fn_context.get_by_col_name('INPUT_IBOCR_RECORD')

  # Get bbox of the provenance tracked field_value
  provenance_info, error = refiner.call_v('provenance_get', field_value, **kwargs)
  prov_info = provenance_info.value()[0]

  x1, y1 = prov_info["image_start"]["x"], prov_info["image_start"]["y"]
  x2, y2 = prov_info["image_end"]["x"], prov_info["image_end"]["y"]
  
  # Get page image
  line_start = prov_info['original_start_2d']['line']
  first_word = record.get_lines()[line_start][0]
  page_num = first_word['page']
  page_image = _get_pil_image_from_ibocr(ibfile, orig_ibocr, page_num)

  ydel = (40,40) # (0,0) # 
  # get cropped image in the form of base64 string
  crop_image = page_image.crop((x1, y1-ydel[0], x2, y2+ydel[1]))
  buffered = io.BytesIO()
  crop_image.save(buffered, format="JPEG")
  output_image = Value(base64.b64encode(buffered.getvalue()).decode())
  pil_image = Image.open(buffered)

  # set the trackers for output image
  tracker = field_value.tracker()
  tracker.convert_to_informational()
  output_image.set_tracker(tracker)
  start = _create_image_position(x1, y1-ydel[0], x2, y2+ydel[1], page_num)
  output_image.set_image_tracker(ImageProvenanceTracker(start))

  return output_image


# --------------------------------------------------------
# --------- Matching signatures ---------------------------
# --------------------------------------------------------

def get_similarity_scores(test_signature: Value[str], ref_signatures: Union[Value[str], List[Value[str]]], clean_test: bool = False, clean_ref: bool = False, rotate: bool = False, **kwargs) -> List[Value[float]]:

    """
    Matches a signature image against reference signature images or a reference folder.
    ```
    Args:
        test_signature (Value[str]): Test signature to be matched against.
        ref_signatures (Union[Value[str], List[Value[str]]]): Either the folder location of reference signature images or a list of reference signatures.
        clean_test (bool, optional): Default is False. A flag indicating whether to clean the test signature or not. 
        clean_ref (bool, optional): Default is False. A flag indicating whether to clean the reference signatures or not. 
        rotate (bool, optional): Default is False, A flag indicating whether to rotate the signature or not.

    Returns:
        List[Value[float]]: Returns a list of similarity scores between the test signature and each reference signature.

    Examples:
        get_similarity_scores(test_signature, '/my-repo/fs/Instabase Drive/files/Signature dataset/user1', true, false)
        -> [0.85, 0.43]
        
        get_similarity_scores(test_signature, match(INPUT_COL, '\[SIGN\]'), false, true)
        -> [0.85, 0.78]
    ```
    """

    if isinstance(ref_signatures.value(), str):
      if rotate:
        test_signature = Value(rotate_image(test_signature.value(), ref_signatures.value(), **kwargs))
      return find_similarity(test_signature, ref_signatures, clean_test, clean_ref, **kwargs)

    fn_context = kwargs.get("_FN_CONTEXT_KEY")
    logger, error = fn_context.get_by_col_name("LOGGER")

    similarity_scores = []

    # Get the test Signature image from the extracted crop
    test_img_bytes = base64.b64decode(test_signature.value())
    if len(test_img_bytes) == 0:
        return [None] * len(ref_signatures)

    # Clean the test Signature image
    if clean_test:
        cleaner_model_result = sign_clean_impl(test_img_bytes, **kwargs)
        if not cleaner_model_result:
            return [None] * len(ref_signatures)
        
        # Get bytes of cleaned image
        cleaned_img_encoded = cleaner_model_result['model_result']['custom_result']['result']
        test_img_bytes = base64.b64decode(cleaned_img_encoded)

    
    # Iterate over each test signature
    for ref_signature in ref_signatures.value():
        # Extract the test signature crop
        extracted_signature = extract_signature_crop(ref_signature, **kwargs)
        if extracted_signature is None:
            similarity_scores.append(None)
            continue
        
        # Get the test Signature image from the extracted crop
        ref_img_bytes = base64.b64decode(extracted_signature.value())
        if len(ref_img_bytes) == 0:
            similarity_scores.append(None)
            continue

        # Clean the test Signature image
        if clean_ref:
            cleaner_model_result = sign_clean_impl(ref_img_bytes, **kwargs)
            if not cleaner_model_result:
                similarity_scores.append(None)
                continue
            
            # Get bytes of cleaned image
            cleaned_img_encoded = cleaner_model_result['model_result']['custom_result']['result']
            ref_img_bytes = base64.b64decode(cleaned_img_encoded)

        # Match the test signature with reference signature
        match_score = sign_verify(ref_img_bytes, test_img_bytes, **kwargs) # Matching cleaned sign with reference sign
        data = json.loads(match_score)
        similarity_val = float(data["similarity_val"])
        similarity_scores.append(round(similarity_val, 2))
    return similarity_scores


# --------------------------------------------------------
# --------- Verifying the signature ---------------------------
# --------------------------------------------------------

def verify_signature(test_signature: Value[str], ref_folder: Value[str], clean_test: bool = False, clean_ref: bool = False, rotate: bool = False, **kwargs) -> Value[str]:

    """ 
    Matches a signature image against reference signature images and determines its authenticity.
    ```
    Args:
        test_signature (Value[str]): An image of the signature to be matched.
        ref_folder (Value[str]): The folder location where reference signature images are stored.
        clean_test (bool, optional): Default is False. A flag indicating whether to clean the test signatures or not. 
        clean_ref (bool, optional): Default is False. A flag indicating whether to clean the reference signature or not.
        rotate (bool, optional): Default is False, A flag indicating whether to rotate the signature or not.
    Returns:
        Value[str]: Returns a text containing the result of the signature matching process, indicating whether the signature is genuine or forged.

    Examples:
        verify_signature(extracted_signature, '/my-repo/fs/Instabase Drive/files/Signature dataset/user4', false)
        -> "Genuine"
    ```
    """

    # rotating the image
    if rotate:
        test_signature = Value(rotate_image(test_signature.value(), ref_folder.value(), **kwargs))

    # (1) Finding the threshold for the images
    user_folder = ref_folder.value()
    threshold, min_similarity = find_threshold(user_folder, **kwargs)

    if threshold == 0:
        return "None"

    # (2) Finding the similarity value between the ref and test images
    similarity_scores = find_similarity(test_signature, ref_folder, clean_test, clean_ref, **kwargs)
    total_score = sum(similarity_scores)

    if total_score == 0:
        return "None" 

    final_score = max(similarity_scores)

    logging.info(f"Final similarity score: {final_score}")
    logging.info(f"Similarities :{similarity_scores}")
    logging.info(f"Threshold : {threshold}")

    if final_score > threshold :
        return "Genuine"
    else:
        return "Forged"
    


# --------------------------------------------------------
# --------- Registering the Functions ---------------------------
# --------------------------------------------------------

def register(name_to_fn):
  name_to_fn.update({
    'extract_signature_crop': {
      'fn': extract_signature_crop,
      'fn_v': extract_signature_crop,
      'ex': '',
      'desc': '',
    },
    'verify_signature' :{
      'fn': verify_signature,
      'fn_v': verify_signature,
      'ex': '',
      'desc': ''
    },
    'get_similarity_scores':{
      'fn': get_similarity_scores,
      'fn_v': get_similarity_scores,
      'ex': '',
      'desc': ''
    }
  })
  