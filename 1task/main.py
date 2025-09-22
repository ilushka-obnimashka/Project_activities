import gradio as gr
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


def main():
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    def process_txt(txt: str) -> dict[str, str | list[str]]:
        return {"text": txt, "entities": ner_pipeline(txt)}

    examples = [
        "Anna has never seen such self-documenting code from an NSU student.",
        "Messenger Max wants to buy Ilya Trushkin's ACMS Censor for a lot of money.",
        "I wonder if they will notice what I wrote here?"
    ]

    demo = gr.Interface(
        fn=process_txt,
        inputs=gr.Textbox(placeholder="Enter a sentence here...", label="Text", lines=3),
        outputs=gr.HighlightedText(label="processed text", show_legend=True),
        title="Project activities. GenAI-1-43(Named Entity Recognition)",
        examples=examples,
        flagging_mode="never",
        theme="soft"
    )

    demo.launch()


if __name__ == '__main__':
    main()
