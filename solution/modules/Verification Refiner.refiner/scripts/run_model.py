import base64
import json
from typing import Any, cast, Dict, List, Optional, Text, Tuple, Union
from mypy_extensions import TypedDict

from ib.market.ib_intelligence.functions import IntelligencePlatform
from instabase.ocr.client.libs.ibocr import ParsedIBOCRBuilder
from instabase.provenance.tracking import Value
from instabase.protos.model_service import model_service_pb2
from instabase.provenance.registration import register_fn

FIELD_RESULT_TYPE_TEXT = "TEXT"
FIELD_RESULT_TYPE_LIST = "LIST"
FIELD_RESULT_TYPE_TEXT_MULTIPLE_INSTANCES = "TEXT_MULTIPLE_INSTANCES"
MODEL_RESULT_TYPE_TABLE = "Table"


class ExtractedWord(TypedDict):
    """
    Represents a single extracted word from a model.
    """

    value: Value[Any]
    confidence: float


class TextFieldExtraction(TypedDict):
    value: Value[List[Value[ExtractedWord]]]
    avg_confidence: Optional[float]
    item_id: Optional[Text]


class ListFieldExtraction(TypedDict):
    value: Value[List[Value[TextFieldExtraction]]]
    avg_confidence: Optional[float]


class ModelResultField(TypedDict):
    field_type: Text
    prediction: Value[Union[ListFieldExtraction, TextFieldExtraction]]


# A mapping of field names to the results for that field
ModelResult = Value[Dict[str, Value[ModelResultField]]]
ModelResultNoProvType = Dict[str, List[List[Union[str, float]]]]
OTHER_TABLES_FIELD_NAME = "other_tables"


def _avg(values: List[Union[float, int]]) -> Optional[float]:
    """
    Calculates the average of the input values, and returns None
    if no values are provided
    """
    values = [v for v in values if v is not None]
    if len(values) == 0:
        return None

    return sum(values) / len(values)


def table_result_to_values(result: Any, INPUT_COL: Value[str]) -> ModelResult:
    r_data = result.get("raw_data", {}).get("data", b"")
    json_result = json.loads(base64.b64decode(r_data).decode("utf-8"))
    fields = json_result["fields"]
    results: Dict[str, Any] = {"result_type": MODEL_RESULT_TYPE_TABLE, "avg_confidence": {}}

    for field in fields:
        field_name = field["field_name"] or OTHER_TABLES_FIELD_NAME

        if field_name not in results["avg_confidence"]:
            results["avg_confidence"][field_name] = []

        if field_name not in results:
            results[field_name] = []

        table_annotations = field["table_annotations"]
        for table_annotation in table_annotations:
            results["avg_confidence"][field_name].append(table_annotation.get("avg_confidence", 0))
            row_len, col_len = len(table_annotation["rows"]), len(table_annotation["cols"])
            cells = [["" for _ in range(col_len)] for i in range(row_len)]
            for cell in table_annotation["cells"]:
                # TODO(qxie3): handle merge cell
                # we currently assume start_index == end_index before supporting merge cell
                # concat list of words
                index_spans: List[Tuple[int, int]] = cell.get("index_spans") or []
                cell_words = []
                for start_index, end_index in index_spans:
                    cell_words.append(INPUT_COL[start_index:end_index])

                cells[cell["row_start_index"]][cell["col_start_index"]] = Value.join(" ", cell_words)

            results[field_name].append(cells)

    field_names = set([f["field_name"] or OTHER_TABLES_FIELD_NAME for f in fields])
    for field_name in field_names:
        if results[field_name]:
            results[field_name] = Value.wrap_value(results[field_name])

    return Value(results)


#########################################################################
# Functions for models returning a model_service_pb2.GenericModelResult
# object as the model result
#########################################################################
def ner_result_to_value(result: Dict, INPUT_COL: Value[str]) -> Value[TextFieldExtraction]:
    item_id = result.get("item_id")

    extracted_words: List[Value[ExtractedWord]] = []
    word_confs: List[float] = []

    # Process all words in this prediction item
    for word in result.get("entities", []):
        extracted_words.append(
            Value(
                ExtractedWord(
                    value=INPUT_COL[word["start_index"] : word["end_index"]],
                    confidence=word.get("score"),
                )
            )
        )
        word_confs.append(word.get("score"))

    return Value(TextFieldExtraction(value=Value(extracted_words), avg_confidence=_avg(word_confs), item_id=item_id))


def ner_result_list_to_value(result: Dict, INPUT_COL: Value[str]) -> Value[ModelResultField]:
    field_type = result["field_type"]

    text_vals: List[Value[TextFieldExtraction]] = [
        ner_result_to_value(item, INPUT_COL) for item in (result.get("items") or [])
    ]
    if (
        model_service_pb2.NERResultList.FieldType.Value(field_type) == model_service_pb2.NERResultList.FieldType.TEXT
    ):  # type: ignore
        item_id = None
        all_words: List[Value[ExtractedWord]] = []
        all_word_confs: List[float] = []
        for text_val in text_vals:
            text_val_unwrapped = text_val.value()
            item_id = text_val_unwrapped["item_id"]

            words = text_val_unwrapped["value"].value()
            all_words.extend(words)
            all_word_confs.extend([word.value().get("confidence") for word in words])

        return Value(
            ModelResultField(
                field_type=FIELD_RESULT_TYPE_TEXT,
                prediction=Value(
                    TextFieldExtraction(value=Value(all_words), avg_confidence=_avg(all_word_confs), item_id=item_id)
                ),
            )
        )
    elif (
        model_service_pb2.NERResultList.FieldType.Value(field_type) == model_service_pb2.NERResultList.FieldType.LIST
    ):  # type: ignore
        return Value(
            ModelResultField(
                field_type=FIELD_RESULT_TYPE_LIST,
                prediction=Value(
                    ListFieldExtraction(
                        value=Value(text_vals),
                        avg_confidence=_avg([text_val.value()["avg_confidence"] for text_val in text_vals]),
                    )
                ),
            )
        )
    elif (
        model_service_pb2.NERResultList.FieldType.Value(field_type)
        == model_service_pb2.NERResultList.FieldType.TEXT_MULTIPLE_INSTANCES
    ):  # type: ignore
        return Value(
            ModelResultField(
                field_type=FIELD_RESULT_TYPE_TEXT_MULTIPLE_INSTANCES,
                prediction=Value(
                    ListFieldExtraction(
                        value=Value(text_vals),
                        avg_confidence=_avg([text_val.value()["avg_confidence"] for text_val in text_vals]),
                    )
                ),
            )
        )
    else:
        raise ValueError(f"Got invalid field type {field_type}")


def generic_model_result_to_value(result: Dict, INPUT_COL: Value[str]) -> ModelResult:
    result_dict: Dict[str, Value[ModelResultField]] = {}
    for field in result.get("generic_model_result", {}).get("fields", []):
        if field.get("ner_result_list"):
            result_dict[field["name"]] = ner_result_list_to_value(field.get("ner_result_list"), INPUT_COL)
        else:
            raise ValueError(f"ner_result_list not present in {result}")

    return Value(result_dict)


######################################################
# Helpers for accessing the avg confidences and values
# for a field in ModelResult
######################################################


def _get_value_for_extracted_word(extracted_word: Value[ExtractedWord]) -> Value[Text]:
    """
    For a given ExtractedWord, return the Value object for the word with populated model confidence. Effectively
    unwraps ExtractedWord into Value[Text] and sets the model confidence.
    """
    val = extracted_word.value().get("value")
    # don't set confidence if the string doesn't have words in it
    if val.value().split() != []:
        val.set_model_confs_for_string([extracted_word.value().get("confidence")])
    return val


def _get_value_for_text_field_extraction(pred: TextFieldExtraction) -> Tuple[Value[Text], float]:
    """
    For a given TextFieldExtraction, return the full string that it
    represents and its confidence
    """
    return (
        Value.join(
            " ",
            [
                # Populate the value object with the model confidence
                _get_value_for_extracted_word(extracted_word)
                for extracted_word in (pred.get("value", Value([])).value() or [])
                if extracted_word.value().get("value", Value(None)).value() is not None
            ],
        ),
        pred.get("avg_confidence"),
    )


def _get_highest_confidence_result_from_list(list_pred: ListFieldExtraction) -> Tuple[Value[Text], Optional[float]]:
    """
    Returns the text and confidence score of the prediction instance in the given
    list with the highest confidence.

    If no instances are found, returns
    (Value(""), None)
    """
    best_match_instance = Value("")
    max_conf = None
    for text_field in cast(Value, list_pred.get("value", Value([]))).value():
        instance, conf = _get_value_for_text_field_extraction(text_field.value())

        if conf is not None and (max_conf is None or conf > max_conf):
            best_match_instance = instance
            max_conf = conf
        elif max_conf is None:
            # If we haven't found a non-null confidence score, make sure
            # we return an instance, instead of the default empty string.
            best_match_instance = instance

    return best_match_instance, max_conf


def get_value_for_field(model_result: ModelResult, field_name: Text) -> Value[Any]:
    model_field_result = model_result.value().get(field_name, Value(None)).value()
    if not model_field_result:
        raise ValueError(f"Field '{field_name}' not found in model result")

    field_type = model_field_result.get("field_type")
    if field_type is None:
        raise ValueError(f"Field type not found for field '{field_name}'")

    pred = model_field_result.get("prediction", Value(None)).value()
    if not pred:
        raise ValueError(f"No prediction found for field '{field_name}' in model result")

    if field_type == FIELD_RESULT_TYPE_TEXT:
        return _get_value_for_text_field_extraction(cast(TextFieldExtraction, pred))[0]

    elif field_type == FIELD_RESULT_TYPE_LIST:
        pred = cast(ListFieldExtraction, pred)
        return Value(
            [
                _get_value_for_text_field_extraction(text_field.value())[0]
                for text_field in cast(Value, pred.get("value", Value([]))).value()
            ]
        )

    elif field_type == FIELD_RESULT_TYPE_TEXT_MULTIPLE_INSTANCES:
        best_match_instance, _ = _get_highest_confidence_result_from_list(pred)
        return best_match_instance
    else:
        raise ValueError(f"Got invalid field type: {field_type}")


@register_fn(provenance=False)
def average(values: List[float], **kwargs) -> Union[float, str]:
    if len(values) == 0:
        return "undefined"
    return sum(values) / len(values)


@register_fn
def get_field(MODEL_RESULT_COL: ModelResult, field_name: Value[str], **kwargs) -> Value[str]:
    """Returns space-joined values of an extracted field

    Equivalent to the following:
      join(' ',
        map(
          map_get(
            MODEL_RESULT_COL,
            field_name,
            default=list()
          ),
          'first(x)'
        )
      )

    Args:
      MODEL_RESULT_COL(dict): dictionary containing model output
      field_name(str): field name

    Returns:
      Returns the extracted values, joined together by whitespace.

    """
    if (
        "result_type" in MODEL_RESULT_COL.value()
        and MODEL_RESULT_COL.value().get("result_type") == MODEL_RESULT_TYPE_TABLE
    ):
        field_value = MODEL_RESULT_COL.value().get(field_name.value())

        if field_value is None:
            return Value.wrap_value([[]])
        # If the field name is OTHER_TABLES_FIELD_NAME,
        # then return all tables. Else, only return the
        # first table
        elif field_name.value() == OTHER_TABLES_FIELD_NAME:
            return field_value
        else:
            return field_value.value()[0]
    else:
        return get_value_for_field(MODEL_RESULT_COL, field_name.value())


# type for error messages
ErrorStr = str


@register_fn(provenance=False, name="get_confidence")
def get_confidence_no_provenance(
    MODEL_RESULT_COL: ModelResultNoProvType, field_name: str, **kwargs
) -> Union[float, ErrorStr]:
    """Returns the confidence score of an extracted field

    Equivalent to the following:
      average(
        map(
          map_get(
            MODEL_RESULT_COL,
            field_name,
            default=list()
          ),
          'last(x)'
        )
      )

    If the field type is FIELD_RESULT_TYPE_TEXT_MULTIPLE_INSTANCES,
    returns the highest confidence score from the list of predictions.

    Args:
      MODEL_RESULT_COL(dict): dictionary containing model output
      field_name(str): field name

    Returns:
      Returns the confidence score, between 0 and 1.

    """
    if "result_type" in MODEL_RESULT_COL and MODEL_RESULT_COL.get("result_type") == MODEL_RESULT_TYPE_TABLE:
        return MODEL_RESULT_COL.get("avg_confidence", {}).get(field_name, [])
    else:
        model_field_result = MODEL_RESULT_COL.get(field_name) or {}
        field_type = model_field_result.get("field_type")
        pred = model_field_result.get("prediction") or {}

        if field_type == FIELD_RESULT_TYPE_TEXT_MULTIPLE_INSTANCES:
            conf = None
            for text_field in pred.get("value") or []:
                curr_conf = text_field.get("avg_confidence")
                if conf is None or (curr_conf is not None and curr_conf > conf):
                    conf = curr_conf
        else:
            conf = pred.get("avg_confidence")

        if conf is None:
            return "Error: no confidences obtained"
        return conf


@register_fn(provenance=False, name="get_field")
def get_field_no_provenance(MODEL_RESULT_COL: ModelResultNoProvType, field_name: str, **kwargs) -> str:
    """Returns space-joined values of an extracted field

    Equivalent to the following:
      join(' ',
        map(
          map_get(
            MODEL_RESULT_COL,
            field_name,
            default=list()
          ),
          'first(x)'
        )
      )

    Args:
      MODEL_RESULT_COL(dict): dictionary containing model output
      field_name(str): field name

    Returns:
      Returns the extracted values, joined together by whitespace.

    """
    raise NotImplementedError("Provenance Tracking must be on to call get_field. Please turn it on in File > Settings.")


@register_fn(provenance=False, name="run_model")
def run_model_no_provenance(INPUT_COL: str, model_version: str, **kwargs) -> Any:
    """
    Runs the trained model associated with this refiner program.
    """
    raise NotImplementedError("Provenance Tracking must be on to run a model. Please turn it on in File > Settings.")


@register_fn
def run_model(
    INPUT_COL: Value[str],
    model_name: Value[str],
    model_version: Value[str],
    is_project_model: Value[str] = Value("False"),
    model_fs_path: Value[str] = Value(""),
    **kwargs,
) -> ModelResult:
    ip_sdk = IntelligencePlatform(kwargs)
    ibocr, err = kwargs["_FN_CONTEXT_KEY"].get_by_col_name("INPUT_IBOCR_RECORD")
    if err:
        raise KeyError(err)
    record = ParsedIBOCRBuilder(use_ibdoc=True)
    record.add_ibocr_records([ibocr])
    results = ip_sdk.run_model(
        model_name.value(),
        input_record=record,
        refresh_registry=False,
        model_version=model_version.value(),
        is_project_model=(is_project_model.value() == str(True)),
        model_fs_path=model_fs_path.value(),
        **kwargs,
    )
    if results.get("generic_model_result"):
        return generic_model_result_to_value(results, INPUT_COL)
    elif "raw_data" in results and results["raw_data"].get("type") == MODEL_RESULT_TYPE_TABLE:
        return table_result_to_values(results, INPUT_COL)
    else:
        raise TypeError("run_model returns unexpected result type.")
