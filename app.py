"""
Arabic Abstractive Text Summarizer — Gradio App
Seq2Seq BiLSTM Encoder + Bahdanau Attention Decoder

Prerequisites:
  pip install gradio tensorflow numpy pandas

Required files in the same directory as this script:
  - arabic_seq2seq_attention_model.keras
  - arabic_tokenizer.pkl
  - preprocessed_data.csv  (optional — enables live dataset examples)
"""

import os
import re
import pickle
import random
import shutil

import numpy as np
import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter

# ── Config — must match training ──────────────────────────────────────────────
VOCAB_SIZE  = 30000
MAX_ART_LEN = 350
MAX_SUM_LEN = 50
LATENT_DIM  = 128
START_TOKEN = 'sostoken'
END_TOKEN   = 'eostoken'
OOV_TOKEN   = '<oov>'

# ── Custom loss required to deserialize the saved model ──────────────────────
def masked_sparse_ce(y_true, y_pred):
    mask           = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss_per_token = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    masked_loss    = loss_per_token * mask
    return tf.reduce_sum(masked_loss) / (tf.reduce_sum(mask) + 1e-8)


# ── Load training model + tokenizer ──────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH     = os.path.join(BASE_DIR, 'arabic_seq2seq_attention_model.keras')
TOKENIZER_PATH = os.path.join(BASE_DIR, 'arabic_tokenizer.pkl')

print("Loading model…")
training_model = load_model(
    MODEL_PATH,
    custom_objects={'masked_sparse_ce': masked_sparse_ce}
)
print("Model loaded.")

with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)
print("Tokenizer loaded.")

# ── Rebuild encoder inference model ──────────────────────────────────────────
enc_input  = training_model.input[0]
bilstm_seq = training_model.get_layer('encoder_bilstm').output[0]
state_h    = training_model.get_layer('encoder_h').output
state_c    = training_model.get_layer('encoder_c').output

encoder_model = tf.keras.Model(
    inputs=enc_input,
    outputs=[bilstm_seq, state_h, state_c],
    name='encoder_inference'
)

# ── Rebuild decoder inference model ──────────────────────────────────────────
encoder_embedding = training_model.get_layer('encoder_embedding')
decoder_lstm      = training_model.get_layer('decoder_lstm')
attention_layer   = training_model.get_layer('bahdanau_attention')
decoder_dense     = training_model.get_layer('word_softmax')

dec_single_in = Input(shape=(1,),                          name='dec_single_in')
enc_out_in    = Input(shape=(MAX_ART_LEN, LATENT_DIM * 2), name='enc_out_in')
state_h_in    = Input(shape=(LATENT_DIM * 2,),             name='state_h_in')
state_c_in    = Input(shape=(LATENT_DIM * 2,),             name='state_c_in')

dec_emb                  = encoder_embedding(dec_single_in)
dec_out, next_h, next_c  = decoder_lstm(dec_emb, initial_state=[state_h_in, state_c_in])
attn_out                 = attention_layer([dec_out, enc_out_in], use_causal_mask=False)
concat_out               = Concatenate(axis=-1)([dec_out, attn_out])
word_probs               = decoder_dense(concat_out)

decoder_model = tf.keras.Model(
    inputs=[dec_single_in, enc_out_in, state_h_in, state_c_in],
    outputs=[word_probs, next_h, next_c],
    name='decoder_inference'
)
print("Inference models ready.")

# ── Vocabulary lookup ─────────────────────────────────────────────────────────
word_index = tokenizer.word_index
index_word = {idx: word for word, idx in word_index.items() if idx < VOCAB_SIZE}
start_id   = word_index[START_TOKEN]
end_id     = word_index[END_TOKEN]

# ── Manual ROUGE (matches notebook implementation) ────────────────────────────
def get_ngrams(tokens, n):
    return Counter(zip(*[tokens[i:] for i in range(n)]))

def rouge_n(reference, prediction, n):
    ref_tokens  = reference.split()
    pred_tokens = prediction.split()
    ref_ngrams  = get_ngrams(ref_tokens,  n)
    pred_ngrams = get_ngrams(pred_tokens, n)
    overlap     = sum((pred_ngrams & ref_ngrams).values())
    precision   = overlap / max(sum(pred_ngrams.values()), 1)
    recall      = overlap / max(sum(ref_ngrams.values()),  1)
    if precision + recall == 0:
        return 0.0, 0.0, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

def rouge_l(reference, prediction):
    ref_tokens  = reference.split()
    pred_tokens = prediction.split()
    r, p        = len(ref_tokens), len(pred_tokens)
    dp = [[0] * (p + 1) for _ in range(r + 1)]
    for i in range(1, r + 1):
        for j in range(1, p + 1):
            if ref_tokens[i-1] == pred_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs       = dp[r][p]
    precision = lcs / max(p, 1)
    recall    = lcs / max(r, 1)
    if precision + recall == 0:
        return 0.0, 0.0, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


# ── Arabic preprocessing (mirrors Phase 1 cleaning) ───────────────────────────
def preprocess_arabic(text):
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^؀-ۿ\s]', ' ', text)
    text = re.sub(r'[أإآٱ]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ؤ', 'و', text)
    text = re.sub(r'[ئى]', 'ي', text)
    text = re.sub(r'[ً-ٰٟ]', '', text)
    text = re.sub(r'ـ', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── Greedy decoding with repetition guard ─────────────────────────────────────
def generate_summary(input_seq, max_repeat=3):
    enc_out, h, c = encoder_model.predict(input_seq, verbose=0)
    target_seq    = np.array([[start_id]])
    summary_words = []
    word_count    = {}

    for step in range(MAX_SUM_LEN):
        probs, h, c = decoder_model.predict(
            [target_seq, enc_out, h, c], verbose=0
        )
        token_probs = probs[0, -1, :]
        sorted_ids  = np.argsort(token_probs)[::-1]

        next_id = next_word = None
        for cid in sorted_ids[:20]:
            cword = index_word.get(int(cid), '')
            if cword in ('', OOV_TOKEN, START_TOKEN, END_TOKEN) or int(cid) == 0:
                continue
            next_id, next_word = int(cid), cword
            break

        if next_id is None:
            break
        if next_id == end_id and step >= 3:
            break

        word_count[next_word] = word_count.get(next_word, 0) + 1
        if word_count[next_word] > max_repeat:
            continue

        summary_words.append(next_word)
        target_seq = np.array([[next_id]])

    return ' '.join(summary_words)


# ── Core Gradio handler ───────────────────────────────────────────────────────
def summarize(article_text, reference_text):
    if not article_text or not article_text.strip():
        return "الرجاء إدخال نص عربي.", ""

    cleaned = preprocess_arabic(article_text)
    if not cleaned:
        return "لم يتم العثور على نص عربي في المدخلات.", ""

    seq    = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_ART_LEN, padding='post', truncating='post')

    summary = generate_summary(padded)
    if not summary:
        return "لم يتمكن النموذج من توليد ملخص. جرب مقالة أطول.", ""

    rouge_md = ""
    if reference_text and reference_text.strip():
        ref_clean     = preprocess_arabic(reference_text)
        r1p, r1r, r1f = rouge_n(ref_clean, summary, 1)
        r2p, r2r, r2f = rouge_n(ref_clean, summary, 2)
        rlp, rlr, rlf = rouge_l(ref_clean, summary)
        rouge_md = (
            f"**ROUGE Scores** (vs. reference)\n\n"
            f"| Metric | Precision | Recall | F1 |\n"
            f"|--------|-----------|--------|----|\n"
            f"| ROUGE-1 | {r1p:.3f} | {r1r:.3f} | {r1f:.3f} |\n"
            f"| ROUGE-2 | {r2p:.3f} | {r2r:.3f} | {r2f:.3f} |\n"
            f"| ROUGE-L | {rlp:.3f} | {rlr:.3f} | {rlf:.3f} |"
        )

    return summary, rouge_md


# ── Clear stale example cache so results always recompute fresh ───────────────
_cache_dir = os.path.join(BASE_DIR, 'gradio_cached_examples')
if os.path.exists(_cache_dir):
    shutil.rmtree(_cache_dir)
    print("Cleared example cache.")

# ── Load examples — random sample from CSV if available, else hardcoded pool ──
# Hardcoded pool: 5 test-set samples from notebook cell 22 qualitative check.
_EXAMPLE_POOL = [
    [
        "مءتمر اسلامي مثير للجدل، عقد كولونيا مطلع العام الجديد ، طالبت حكومه ولايه الراين "
        "ويستفاليا، اكبر ولايه المانيه عدد السكان، فيهم المسلمين، اتحاد المساجد التركيه "
        "الالمانيه ديتيب بالتحرر التبعيه للدوله التركيه والابتعاد تاثيرها المباشر وقال متحدث "
        "باسم حكومه الراين ويستفاليا لصحيفه غنرال انتسايغر بون",
        "مءتمر مثير للجدل حول الاسلام المانيا عقد مدينه كولونيا مطلع الشهر الجاري، طالبت "
        "حكومه الراين ويستفاليا اتحاد المساجد الاسلاميه التركيه ديتيب بالتحرر التبعيه للدوله "
        "التركيه والخروج نفوذها المباشر",
    ],
    [
        "قال المتحدث باسم الحكومه الالمانيه، شتيفن زايبرت، اليوم الاربعاء الاول اذار بالعاصمه "
        "برلين نخطط الوقت الراهن لانشاء ايه مراكز علي الاطلاق وشدد علي ضروره ان يحصل خبراء "
        "المفوضيه العليا لشءون اللاجءين التابعه للامم المتحده علي امكانيه للوصول الي مرافق "
        "ليبيا اجل تحقيق تعامل افضل للاشخاص يعيشون هناك",
        "اعلن مرشح الحزب الاشتراكي الديمقراطي للمنافسه علي منصب المستشاريه مارتن شولتس، رفضه "
        "لاقامه مراكز استقبال لاجءين افريقيا ابدي زعيم الحزب البافاري تمسكه بوضع حد اقصي "
        "للاجءين القادمين الي المانيا",
    ],
    [
        "الانتخابات التونسيه النزيهه ابرزت قوه الطرح الاسلامي السياسي واصطفاف جزء عريض الشعب "
        "حوله انعكس علي الوضع المغرب وخلق حاله الخوف لدي منافسي حزب العداله والتنميه الاسلامي "
        "اكتساح الاخير للانتخابات المقبله بهذه الكلمات يلخص ياسين علمي، طالب الهندسه، متحدثا "
        "لدويتشه فيله، موقفه فوز حزب النهضه التونسي",
        "يسود التفاءل حزب العداله والتنميه الاسلامي المغربي بتحقيق فوز مماثل لحزب النهضه "
        "التونسي الانتخابات تجري الشهر والشارع المغربي يبدو منقسما ازاء الفرضيه، ومراقبون "
        "يشيرون الي اختلافات تجربتي البلدين",
    ],
    [
        "راغب عياد اواءل الفنانين المصريين اللي اهتموا بتصوير الحياه اليوميه المصريه الاصيله، "
        "خاصه الريف اعماله كانت بتتميز بالواقعيه والتركيز علي التفاصيل الدقيقه للمشاهد الشعبيه "
        "والفلاحين والحيوانات سافر عياد لايطاليا للدراسه وده اثر علي اسلوبه استخدام الالوان "
        "والضوء، لكنه فضل دايما يعبر البيءه المصريه بكل تفاصيلها",
        "راغب عياد، راءد تصوير الحياه المصريه الريفيه بواقعيه ودقه درس ايطاليا لكنه حافظ علي "
        "التعبير البيءه المصريه الاصيله اعماله",
    ],
    [
        "تمت، عشيه اليوم الجمعه ، احاله محام علي الداءره الجناءيه بقرمباليه اجل جريمه التدليس "
        "وتقليد طوابع تابعه للقباضه الماليه وكانت فرقه الشرطه العدليه بقرمباليه ولايه نابل "
        "تمكنت اليوم القاء القبض علي المحامي وحجزت لديه طوابع جباءيه تقدر قيمتها بحوالي مليار "
        "المليمات، بحسب اكده مصدر امني",
        "تمت، عشيه اليوم الجمعه ، احاله محام علي الداءره الجناءيه بقرمباليه اجل جريمه التدليس "
        "وتقليد طوابع تابعه للقباضه الماليه",
    ],
]

EXAMPLES = []
_csv_path = os.path.join(BASE_DIR, 'preprocessed_data.csv')
if os.path.exists(_csv_path):
    try:
        import pandas as pd
        _df = pd.read_csv(_csv_path, encoding='utf-8-sig')
        _df = _df.dropna(subset=['article_clean', 'summary_clean'])
        _df = _df[_df['article_clean'].str.strip().astype(bool)]
        _df = _df[_df['summary_clean'].str.strip().astype(bool)].reset_index(drop=True)
        _sample = _df.sample(n=min(3, len(_df)))
        for _, row in _sample.iterrows():
            EXAMPLES.append([row['article_clean'], row['summary_clean']])
        print(f"Loaded {len(EXAMPLES)} random examples from preprocessed_data.csv.")
    except Exception as e:
        print(f"Could not load CSV ({e}), using hardcoded examples.")

if not EXAMPLES:
    EXAMPLES = random.sample(_EXAMPLE_POOL, 3)
    print(f"Using 3 random hardcoded examples.")


# ── Gradio UI ─────────────────────────────────────────────────────────────────
CSS = """
.rtl textarea, .rtl input { direction: rtl; text-align: right; font-family: 'Segoe UI', Tahoma, sans-serif; font-size: 16px; }
.center { text-align: center; }
footer { display: none !important; }
"""

with gr.Blocks(
    title="Arabic Text Summarizer",
    theme=gr.themes.Soft(primary_hue="blue"),
    css=CSS
) as demo:

    gr.Markdown(
        """
        # Arabic Abstractive Text Summarizer
        **Seq2Seq BiLSTM Encoder + Bahdanau Attention Decoder**

        Paste any Arabic news article and click **Summarize** to generate an abstractive summary.
        A reference summary is required to compute ROUGE scores.
        """,
        elem_classes="center"
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            article_box = gr.Textbox(
                label="Arabic Article — المقالة العربية",
                placeholder="الصق المقال العربي هنا…",
                lines=12,
                elem_classes="rtl",
            )
            reference_box = gr.Textbox(
                label="Reference Summary — الملخص المرجعي",
                placeholder="الصق الملخص المرجعي هنا لحساب درجات ROUGE…",
                lines=4,
                elem_classes="rtl",
            )
            with gr.Row():
                clear_btn     = gr.Button("Clear", variant="secondary")
                summarize_btn = gr.Button("Summarize — لخّص", variant="primary")

        with gr.Column(scale=1):
            summary_box = gr.Textbox(
                label="Generated Summary — الملخص المولَّد",
                lines=6,
                interactive=False,
                elem_classes="rtl",
                show_copy_button=True,
            )
            rouge_box = gr.Markdown()
            gr.Markdown(
                """
                **Model details**
                - Encoder: Bidirectional LSTM (128 units/dir → 256-dim state)
                - Decoder: LSTM (256-dim) + Bahdanau Attention
                - Vocabulary: 30,000 Arabic tokens
                - Max article length: 350 tokens · Max summary: 50 tokens
                - Test-set ROUGE-1: 0.083 · ROUGE-2: 0.014 · ROUGE-L: 0.070
                """
            )

    gr.Examples(
        examples=EXAMPLES,
        inputs=[article_box, reference_box],
        outputs=[summary_box, rouge_box],
        fn=summarize,
        cache_examples="lazy",
        label="Examples from dataset — أمثلة من البيانات",
        examples_per_page=3,
    )

    summarize_btn.click(
        fn=summarize,
        inputs=[article_box, reference_box],
        outputs=[summary_box, rouge_box],
    )
    clear_btn.click(
        fn=lambda: ("", "", ""),
        outputs=[article_box, reference_box, summary_box],
    )

if __name__ == '__main__':
    demo.launch()
