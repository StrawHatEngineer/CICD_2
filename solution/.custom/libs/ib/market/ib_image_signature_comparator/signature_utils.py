import base64
import io
import json
from PIL import Image
import base64
import logging
import cv2
import numpy as np

from typing import Any, Callable, List

from google.protobuf import json_format
from instabase.protos.doc_types import core_types_pb2
from instabase.protos.model_service import model_service_pb2
from instabase.model_service.sdk import ModelService
from instabase.provenance.tracking import Value, ProvenanceBoundingBox, TextCoordinate, ImageProvenanceTracker, ImagePosition

to_json: Callable = lambda proto: json.loads(
  json_format.MessageToJson(proto, preserving_proto_field_name=True))


def _create_image_position(x1, y1, x2, y2, page_num):
  # Internal Function
  
  """
  Convert a bounding box set by {page_num=(page number in record), (x1,
  y1)=top-left coordinate, (x2, y2)=bottom-right coordinate} into an
  ImagePosition object
  """
  start = TextCoordinate(x1, y1)
  bb = ProvenanceBoundingBox(start, x2 - x1, y2 - y1)
  return ImagePosition(bb, page_num)

def _get_pil_image_from_ibocr(ibfile, orig_ibocr, page_num):
  # Internal Function

  """ Get the pil image from the input original ibocr. """
  page_record, err = orig_ibocr.get_ibocr_record(
      page_num if page_num < orig_ibocr.get_num_records() else 0)
  image_path = page_record.get_layout().get_processed_image_path()
  with ibfile.open(image_path, 'rb') as fptr:
    image_bytes = io.BytesIO(fptr.read())
  return Image.open(image_bytes)



# --------------------------------------------------------
# --------- Model calls -----------------------
# --------------------------------------------------------

def sign_verify(ref_img_bytes, test_img_bytes, **kwargs):
  # internal function

  fn_context = kwargs.get("_FN_CONTEXT_KEY")

  ibfile = fn_context.get_ibfile()
  ms = ModelService(username="lalit")

  test_document = core_types_pb2.Document(
      type=core_types_pb2.Document.IMAGE, content = test_img_bytes
  )
  ref_document = {'data': ref_img_bytes}

  model_version = '0.0.2' 
  req = model_service_pb2.RunModelRequest(
      context=ms._get_context(),
      model_name="SignVer",
      model_version=model_version,
      input_document = test_document,
      input_raw_data = ref_document
  )

  # Running model
  try:
    response = ms.run(req)
  except Exception:
    return None

  result = to_json(response)
  return result['model_result']['custom_result']['result']

def sign_clean_impl(test_img_bytes, **kwargs):
  # internal function

  fn_context = kwargs.get("_FN_CONTEXT_KEY")
  logger, error = fn_context.get_by_col_name("LOGGER")

  ibfile = fn_context.get_ibfile()
  ms = ModelService(username="lalit")
  
  test_document = core_types_pb2.Document(
      type=core_types_pb2.Document.IMAGE, content = test_img_bytes
  )

  model_version = '0.0.2'
  req = model_service_pb2.RunModelRequest(
      context=ms._get_context(),
      model_name="SignCleaner",
      model_version=model_version,
      model_service_context = model_service_pb2.ModelServiceContext(force_reload=True),
      input_document = test_document
  )
  
  # Running model
  try:
    response = ms.run(req)
  except Exception:
    return None #, None

  result = to_json(response)
  return result


# --------------------------------------------------------
# --------- Find the threshold ---------------------------
# --------------------------------------------------------

def find_threshold(user_folder, **kwargs):
  # internal function

  fn_context = kwargs.get("_FN_CONTEXT_KEY")
  logger, error = fn_context.get_by_col_name("LOGGER")

  # (1) getting the genuine signatures
  ibfile = fn_context.get_ibfile()
  genuine_signatures = []
  if ibfile.is_dir(user_folder):
    output, err = ibfile.list_dir(user_folder, '')
    nodes = output.nodes
    for filename in nodes:
      image_path = filename.full_path
      if ibfile.is_file(image_path):
        signature_image = ibfile.open('/{}'.format(image_path), 'rb').read()
        genuine_signatures.append(signature_image)
      else:
        logger.info("error finding file !")
        return [0, 0]
  else:
    logger.info(f"no reference user detected!") 
    return [0, 0]

  # (2) comparing genuine signatures and finding the similarity scores
  similarity_scores = []
  if(len(genuine_signatures) <= 1):
    return 80.0

  for x in genuine_signatures: 
    if x != genuine_signatures[0]:
      match_score = sign_verify(genuine_signatures[0], x, **kwargs) # matching extracted sign with reference sign
      data = json.loads(match_score)
      similarity_val = float(data["similarity_val"])
      similarity_scores.append(round(similarity_val, 2))

  # (3) setting the threshold
  target_fnr = 0.70
  sorted_scores = sorted(similarity_scores, reverse=True)
  threshold_index = int(len(sorted_scores) * (1 - target_fnr))
  threshold = sorted_scores[threshold_index]
  return threshold, sorted_scores[-1]


# --------------------------------------------------------
# --------- Find similarity Score ---------------------------
# --------------------------------------------------------

def find_similarity(test_signature: Value[str], ref_folder: Value[str], clean_test: bool = False, clean_ref: bool = False, **kwargs) -> List[Value[float]]:
  # internal function

  fn_context = kwargs.get("_FN_CONTEXT_KEY")
  logger, error = fn_context.get_by_col_name("LOGGER")

  # (1) Get the test Signature image from the given field
  test_img_bytes = base64.b64decode((test_signature.value()))
  if len(test_img_bytes) == 0:
    return None

  # (2) Clean the test Signature image
  if clean_test:
    cleaner_model_result = sign_clean_impl(test_img_bytes, **kwargs)
    if not cleaner_model_result:
      return None
  
    # get bytes of cleaned image
    cleaned_img_encoded = cleaner_model_result['model_result']['custom_result']['result']
    test_img_bytes = base64.b64decode(cleaned_img_encoded)

  # (4) Finding the similarity value between the ref and test images
  user_folder = ref_folder.value()
  ibfile = fn_context.get_ibfile()
  similarity_scores = []
  if ibfile.is_dir(user_folder):
    output, err = ibfile.list_dir(user_folder, '')
    nodes = output.nodes
    for filename in nodes:
      image_path = filename.full_path
      if ibfile.is_file(image_path):
        ref_img_bytes = ibfile.open('/{}'.format(image_path), 'rb').read()

        # Clean the test Signature image
        if clean_ref:
            cleaner_model_result = sign_clean_impl(ref_img_bytes, **kwargs)
            if not cleaner_model_result:
                similarity_scores.append(None)
                continue
            
            # Get bytes of cleaned image
            cleaned_img_encoded = cleaner_model_result['model_result']['custom_result']['result']
            ref_img_bytes = base64.b64decode(cleaned_img_encoded)

        match_score = sign_verify(ref_img_bytes, test_img_bytes, **kwargs) # matching extracted sign with reference sign
        data = json.loads(match_score)
        similarity_val = float(data["similarity_val"])
        similarity_scores.append(round(similarity_val, 2))
      else:
        logger.info("error finding file !")   
        return([None]) 
  else:
    logger.info(f"no reference user detected!") 
    return([None])

  return similarity_scores


# --------------------------------------------------------
# --------- Rotate Image ---------------------------
# --------------------------------------------------------

def load_image_from_folder(test_img_b64: str, user_folder: str, **kwargs) -> bytes:

    fn_context = kwargs.get("_FN_CONTEXT_KEY")
    ibfile = fn_context.get_ibfile()

    test_img_bytes = base64.b64decode(test_img_b64)
    test_img = Image.open(io.BytesIO(test_img_bytes))

    if ibfile.is_dir(user_folder):
        output, err = ibfile.list_dir(user_folder, '')
        nodes = output.nodes

        max_similarity = 0
        best_ref_image_bytes = None

        for filename in nodes:
            image_path = filename.full_path
            if ibfile.is_file(image_path):
                ref_img_bytes = ibfile.open('/{}'.format(image_path), 'rb').read()

                match_score = sign_verify(test_img_bytes, ref_img_bytes, **kwargs)
                data = json.loads(match_score)
                similarity_val = float(data["similarity_val"])

                if similarity_val > max_similarity:
                    max_similarity = similarity_val
                    best_ref_image_bytes = ref_img_bytes
    return best_ref_image_bytes

def check_image_orientation(image_b64):
    image_bytes = base64.b64decode(image_b64)
    image_np = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    height, width = image_np.shape[:2]
    
    # Check orientation
    if width >= height:
        return "horizontal"
    else:
        return "vertical"

def rotate_image_base64(input_img_b64: str, angle: float, **kwargs) -> str:
    input_img_bytes = base64.b64decode(input_img_b64)
    input_img = cv2.imdecode(np.frombuffer(input_img_bytes, np.uint8), cv2.IMREAD_COLOR)

    # Determine the orientation of the input image
    orientation = check_image_orientation(input_img_b64)

    # Rotate the image based on orientation
    if orientation == "horizontal":
        image_bytes = base64.b64decode(input_img_b64)
        input_img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        imgHeight, imgWidth = input_img.shape[:2]
        centreY, centreX = imgHeight // 2, imgWidth // 2
        
        rotationMatrix = cv2.getRotationMatrix2D((centreX, centreY), angle, 1.0)
        rotating_image = cv2.warpAffine(input_img, rotationMatrix, (imgWidth, imgHeight), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        
        _, buffer = cv2.imencode('.jpg', rotating_image)
        rotated_img_b64 = base64.b64encode(buffer).decode()
        
        return rotated_img_b64
    else:
        image_bytes = base64.b64decode(input_img_b64)
        pil_image = Image.open(io.BytesIO(image_bytes))
        rotated_image = pil_image.rotate(angle, expand=True)
        buffered = io.BytesIO()
        rotated_image.save(buffered, format="JPEG")
        rotated_image_bytes = buffered.getvalue()
        rotated_img_b64 = base64.b64encode(rotated_image_bytes).decode()
        return rotated_img_b64

def rotate_image(test_img_b64: str, ref_folder: str, angle_range: int = 360, angle_step: int = 10, **kwargs) -> str:
    ref_img_bytes = load_image_from_folder(test_img_b64, ref_folder, **kwargs)
    if ref_img_bytes is None:
        raise ValueError("No valid reference images found in the folder")

    test_img_bytes = base64.b64decode(test_img_b64)
    test_img = Image.open(io.BytesIO(test_img_bytes))
    best_similarity = 0
    best_image = test_img

    for angle in range(0, angle_range, angle_step):
        rotated_img_b64 = rotate_image_base64(test_img_b64, angle, **kwargs)
        rotated_img_bytes = base64.b64decode(rotated_img_b64)

        match_score = sign_verify(rotated_img_bytes, ref_img_bytes, **kwargs)
        data = json.loads(match_score)
        similarity_val = float(data["similarity_val"])
 
        if similarity_val > best_similarity:
            best_similarity = similarity_val
            best_image = rotated_img_b64
            
    return best_image
