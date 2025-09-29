import sys
import signal
import json

import gradio as gr
from transformers import pipeline, Pipeline

YELLOW = '\033[33m'
RED = "\033[91m"
RESET = "\033[0m"

EXAMPLES = [
    "Anna has never seen such self-documenting code from an NSU student.",
    "Messenger Max wants to buy Ilya Trushkin's ACMS Censor for a lot of money.",
    "I wonder if they will notice what I wrote here?"
]


def keyboard_interrupt_handler(sig, frame):
    """Signal handler for Ctrl-C interrupts.
    See also :
        'signal.py'
    """
    sys.exit()


def get_pipeline(task: str, **kwargs):
    """Return pipeline from Hugging Face Hub with error handling.
    Args:
        task : The task defining which pipeline will be returned. See 'transformers.pipeline()' for more details
        **kwargs: Additional arguments for 'transformers.pipeline()'
    See also:
        'transformers.pipeline()'
    Examples:
    ```python
        >>> ner_pipeline = get_pipeline("ner", model = model)
    ```"""
    try:
        return pipeline(task, **kwargs)
    except Exception as e:
        print(e)
        return -1


def validate_max_length(max_length : int, pipeline : Pipeline) -> int:
    """Validate max_length parameters for text processing."""
    if max_length is None or max_length <= 0:
        length_limit = pipeline.model.config.max_position_embeddings
        print(f"{YELLOW}max_length is invalid ({max_length}). Using model limit: {length_limit}{RESET}")
        max_length = length_limit
    elif max_length > pipeline.model.config.max_position_embeddings:
        length_limit = pipeline.model.config.max_position_embeddings
        print(f"{YELLOW}max_length ({max_length}) exceeds model limit ({length_limit}). Using model limit.{RESET}")
        max_length = length_limit
    return max_length


def validate_overlap(overlap : int, max_length : int) -> int:
    """Validate overlap parameters for text processing."""
    if overlap < 0:
        safe_overlap = min(8, max_length - 1)
        print(f"{YELLOW}overlap cannot be negative ({overlap}). Using: {safe_overlap}{RESET}")
        overlap = safe_overlap
    elif overlap >= max_length:
        safe_overlap = min(8, max_length - 1)
        print(f"{YELLOW}overlap ({overlap}) must be less than max_length ({max_length}). Using: {safe_overlap}{RESET}")
        overlap = safe_overlap
    return overlap


def process_text(tokenizer, pipeline, text, max_length=512, overlap=8):
    """
    Processing text with Transformers.pipeline() using tokenizer offsets (text of any length can be processed).
    Args:
        tokenizer: The tokenizer on Hugging Face Hub
        pipeline: The pipeline on Hugging Face Hub
        text: The text to be processed
        max_length: Maximum chunk length (in tokens) that the model can process in one pass
        overlap: Overlap size between consecutive text chunks

    See also:
        'transformers.pipeline()'
    """
    max_length = validate_max_length(max_length, pipeline)
    overlap = validate_overlap(overlap, max_length)
    try:
        inputs = tokenizer(text, return_tensors='pt', truncation=False,
                           add_special_tokens=False, return_offsets_mapping=True)

        tokens = inputs['input_ids'][0]
        offsets = inputs['offset_mapping'][0]

        all_entities = []
        seen_entities = set()

        for i in range(0, len(tokens), max_length - overlap):
            chunk_offsets = offsets[i:i + max_length]

            chunk_start = chunk_offsets[0][0].item() if len(chunk_offsets) > 0 else 0
            chunk_end = chunk_offsets[-1][1].item() if len(chunk_offsets) > 0 else len(text)

            chunk_text = text[chunk_start:chunk_end]

            if not chunk_text.strip():
                continue

            chunk_entities = pipeline(chunk_text)

            for entity in chunk_entities:
                entity_start = entity['start'] + chunk_start
                entity_end = entity['end'] + chunk_start

                if entity_end > len(text):
                    continue

                entity_word = text[entity_start:entity_end]
                entity_key = (entity_word.lower(), entity_start, entity_end)

                if entity_key not in seen_entities:
                    seen_entities.add(entity_key)
                    entity.update({
                        'start': entity_start,
                        'end': entity_end,
                        'word': entity_word
                    })
                    all_entities.append(entity)
            continue

    except Exception as e:
        exception_type = type(e).__name__
        func_name = process_text.__name__
        print(f"{RED}[{exception_type}] in {func_name}: {e}{RESET}")
        raise e

    return {"text": text, "entities": all_entities}


def _save_to_json(data):
    filename = "result.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return filename


def main():
    signal.signal(signal.SIGINT, keyboard_interrupt_handler)

    try:
        ner_pipeline = get_pipeline("ner", model="dslim/bert-large-NER", aggregation_strategy="simple")
        if ner_pipeline is None:
            print("Failed to create pipeline")
            return -1

        with gr.Blocks(theme="soft") as demo:
            txt_input = gr.Textbox(placeholder="Enter a sentence here...", label="Text", lines=3)
            examples = gr.Examples(examples=EXAMPLES, inputs=txt_input)
            highlighted_output = gr.HighlightedText(label="processed text")
            process_btn = gr.Button("Обработать")
            download_btn = gr.DownloadButton(label="Скачать результат")

            process_btn.click(
                fn=lambda txt: process_text(ner_pipeline.tokenizer, ner_pipeline, txt),
                inputs=txt_input,
                outputs=highlighted_output,
            )
            download_btn.click(
                fn=_save_to_json,
                inputs=highlighted_output,
                outputs=download_btn,
            )
        demo.launch()

    except (KeyboardInterrupt, SystemExit):
        print("\nProgram terminated by user")
        return 0
    except Exception as e:
        print(e)
        return -1


if __name__ == '__main__':
    main()
