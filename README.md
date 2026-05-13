# Arabic Abstractive Text Summarization

An end-to-end NLP project that automatically summarizes Arabic news articles. Built with a Seq2Seq BiLSTM encoder and Bahdanau attention decoder, trained from scratch.

## What it does

Takes an Arabic news article and generates a short summary. You can also paste in a reference summary to get ROUGE scores and see how close the model got.

## Files

| File | Description |
|------|-------------|
| `app.py` | Gradio web app — run this to use the model |
| `NLP_phase2_proj.ipynb` | Full training notebook (preprocessing, model, evaluation) |
| `NLP_Final_Report.docx` | Written report |
| `NLP_Final_Presentation.pptx` | Presentation slides |
| `Final Project_Text Summarization.pdf` | Project brief |

## Running the app

```bash
pip install gradio tensorflow numpy pandas
python app.py
```

Then open the local URL it prints. The app loads a few random examples from the dataset — click one to auto-fill the article and reference summary, then hit **Summarize**.

## Model details

- Architecture: Seq2Seq with BiLSTM encoder + LSTM decoder + Bahdanau attention
- Vocabulary: 30,000 tokens
- Max input length: 350 tokens, max output: 50 tokens
- Trained on Arabic news articles and their human-written summaries
- Evaluation: ROUGE-1, ROUGE-2, ROUGE-L (F1)

## Requirements

- Python 3.8+
- TensorFlow 2.x
- Gradio
- NumPy, Pandas
