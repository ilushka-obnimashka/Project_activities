import gradio as gr
from transformers import pipeline


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
        return None


def process_text(tokenizer, pipeline, text, max_length=512, stride=8):
    """
    Processing text with Transformers.pipeline() using tokenizer offsets (text of any length can be processed).
    Args:
        tokenizer: The tokenizer on Hugging Face Hub
        pipeline: The pipeline on Hugging Face Hub
        text: The text to be processed
        max_length: Maximum chunk length (in tokens) that the model can process in one pass
        stride: Overlap size between consecutive text chunks

    See also:
        'transformers.pipeline()'
    """
    try:
        inputs = tokenizer(text, return_tensors='pt', truncation=False,
                           add_special_tokens=False, return_offsets_mapping=True)

        tokens = inputs['input_ids'][0]
        offsets = inputs['offset_mapping'][0]

        all_entities = []
        seen_entities = set()

        for i in range(0, len(tokens), max_length - stride):
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
        print(f"Error processing chunk: {e}")

    return {"text": text, "entities": all_entities}


def main():
    try:
        ner_pipeline = get_pipeline("ner", model="dslim/bert-large-NER", aggregation_strategy="simple")
        if ner_pipeline is None:
            print("Failed to create pipeline")
            return -1

        examples = [
            "Anna has never seen such self-documenting code from an NSU student.",
            "Messenger Max wants to buy Ilya Trushkin's ACMS Censor for a lot of money.",
            "I wonder if they will notice what I wrote here?"
        ]

        demo = gr.Interface(
            fn=lambda txt: process_text(ner_pipeline.tokenizer, ner_pipeline, txt),
            inputs=gr.Textbox(placeholder="Enter a sentence here...", label="Text", lines=3),
            outputs=gr.HighlightedText(label="processed text"),
            title="Project activities. GenAI-1-43(Named Entity Recognition)",
            examples=examples,
            flagging_mode="never",
            theme="soft"
        )

        demo.launch()

        return 0

    except KeyboardInterrupt:
        print("\nProgram terminated by user")
        return 0


if __name__ == '__main__':
    main()
