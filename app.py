import streamlit as st
import pandas as pd
import joblib
import pickle
import jieba
import re
import io

# 自定义 CSS 样式
st.markdown(
    """
    <style>
        /* 页面背景 */
        body {
            background-color: #f9f9f9;
        }
        /* 单选按钮美化 */
        .stRadio input[type="radio"] {
            accent-color: #FF5733; /* 改变选中时的颜色 */
        }
        .stRadio label {
            font-size: 16px; /* 调整字体大小 */
            color: #333; /* 调整文字颜色 */
        }
        .stRadio input[type="radio"]:checked + label {
            color: #FF5733; /* 选中时的文字颜色 */
        }
        /* 输入框美化 */
        .stTextInput textarea {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 10px;
            font-size: 14px;
        }
        /* 文件上传器美化 */
        .stFileUploader div > div > input {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 10px;
            font-size: 14px;
        }
        /* 数据表格美化 */
        .stDataFrame table {
            font-size: 14px;
            border-collapse: collapse;
            width: 100%;
        }
        .stDataFrame th, .stDataFrame td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        .stDataFrame th {
            background-color: #f2f2f2;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

TFIDF_PICKLE = 'tfidf_vectorizer.pkl'
MODEL_PICKLE = 'ensemble_news_model.pkl'
STOPWORDS_FILE = 'cnews.vocab.txt'

@st.cache_resource  # 使用 st.cache_resource 缓存资源对象
def load_vectorizer():
    with open(TFIDF_PICKLE, 'rb') as f:
        return pickle.load(f)

@st.cache_resource  # 使用 st.cache_resource 缓存资源对象
def load_model():
    return joblib.load(MODEL_PICKLE)

@st.cache_data  # 使用 st.cache_data 缓存停用词集合
def load_stopwords():
    stopwords = set()
    try:
        with open(STOPWORDS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word:
                    stopwords.add(word)
    except Exception as e:
        st.error("停用词加载错误: " + str(e))
    return stopwords

def clean_text(text):
    text = re.sub(r'[^\u4e00-\u9fa5]', ' ', str(text))  # 保留中文字符
    return text.strip()  # 去除首尾空白字符

def tokenize_text(text, stopwords):
    text = clean_text(text)
    if not text:  # 如果清理后为空字符串，则返回空
        return ""
    tokens = jieba.lcut(text)
    tokens = [tok for tok in tokens if tok.strip() and tok not in stopwords]
    return " ".join(tokens)

def classify_texts(texts, stopwords, vectorizer, model):
    processed = []
    non_empty_indices = []  # 记录非空行的索引
    for i, text in enumerate(texts):
        cleaned_text = clean_text(text)
        if cleaned_text:  # 跳过空白行
            processed.append(tokenize_text(cleaned_text, stopwords))
            non_empty_indices.append(i)

    if not processed:  # 如果所有行都是空白行
        return []

    X_input = vectorizer.transform(processed)
    preds = model.predict(X_input)

    # 将预测结果映射回原始文本位置（包括空白行）
    full_preds = [""] * len(texts)  # 初始化全为空字符串
    for idx, pred in zip(non_empty_indices, preds):
        full_preds[idx] = pred

    return full_preds

def main():
    st.title("✨ 中文新闻分类系统 ✨")
    st.markdown("""
    您可以选择输入一段或多段文本，或上传包含新闻文本的 `.txt` 或 `.csv` 文件，  
    系统将为您分类每条文本，并显示结果。
    """)

    # 用户选择输入方式
    input_mode = st.radio(
        "请选择输入方式：",
        ["输入文本", "上传文件"],
        key="input_mode_radio",
        horizontal=True  # 水平排列单选按钮
    )

    stopwords = load_stopwords()
    vectorizer = load_vectorizer()
    model = load_model()

    if input_mode == "输入文本":
        # 多段文本输入
        # st.markdown("请输入多段新闻文本，每段文本用换行符分隔：")
        text_input = st.text_area(
            "",
            placeholder="例如：\n今天的天气非常好\n昨天股市大跌",
            height=150
        )
        if st.button("开始分类"):
            if text_input.strip():  # 检查是否为空
                # 按换行符分割输入内容
                texts = text_input.splitlines()
                predictions = classify_texts(texts, stopwords, vectorizer, model)
                
                # 展示结果
                result_df = pd.DataFrame({
                    "输入文本": texts,
                    "分类结果": predictions
                })
                st.success("🎉 分类完成！以下是结果：")
                st.dataframe(result_df, use_container_width=True)
            else:
                st.warning("请输入有效的文本内容！")

    elif input_mode == "上传文件":
        # 文件上传
        file = st.file_uploader(
            "选择文件上传",
            type=['txt', 'csv'],
            accept_multiple_files=False
        )

        if file is not None:
            try:
                # 初始化 df，默认为空 DataFrame
                df = pd.DataFrame()

                # 自动判断文件格式
                if file.name.endswith('.csv'):
                    df = pd.read_csv(file)
                    st.write("检测到 CSV 文件")
                    # 尝试识别文本列
                    if df.empty:
                        st.error("CSV 文件为空，请上传有效文件！")
                        return
                    text_col = st.selectbox("请选择文本列：", df.columns)
                    texts = df[text_col].fillna("").tolist()  # 替换 NaN 为 ""
                else:
                    st.write("检测到 TXT 文件，每行为一条新闻")
                    texts = file.read().decode('utf-8').splitlines()
                    df = pd.DataFrame({'text': texts})

                # 确保有数据
                if df.empty or not texts:
                    st.error("文件内容为空，请上传有效文件！")
                    return

                # 批量分类
                predictions = classify_texts(texts, stopwords, vectorizer, model)
                df['predicted_label'] = predictions

                st.success("🎉 分类完成！以下是部分结果：")
                st.dataframe(df.head(10), use_container_width=True)

                # 下载链接
                output = io.StringIO()
                df.to_csv(output, index=False, encoding='utf-8')
                st.download_button(
                    label="点击下载分类结果CSV",
                    data=output.getvalue(),
                    file_name='classified_news.csv',
                    mime='text/csv'
                )

            except Exception as e:
                st.error(f"处理文件出错：{e}")

if __name__ == '__main__':
    main()