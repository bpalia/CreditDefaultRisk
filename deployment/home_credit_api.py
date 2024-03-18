from fastapi import FastAPI
import pandas as pd
import gradio as gr
import joblib


app = FastAPI()
model = joblib.load("tuned_100model_full.joblib")


@app.get("/")
def home():
    return "Credit Default Risk Prediction App at /app"


def predict(table: gr.Dataframe, threshold: float, evt: gr.SelectData):
    row = pd.DataFrame([table.iloc[evt.index[0], :]])
    row = row.apply(pd.to_numeric, errors="coerce")
    proba = model.predict_proba(row)[0][1]
    if proba >= threshold:
        prediction = "Potential Defaulter"
    else:
        prediction = "Non-defaulter"
    return proba, prediction


def upload_csv(file_path):
    df = pd.read_csv(file_path, index_col=0)
    return df


with gr.Blocks() as demo:
    gr.Markdown("# Credit Default Risk Prediction App")
    upload_button = gr.UploadButton(
        label="Upload Prepared Application Dataset",
        file_types=[".csv"],
        file_count="single",
    )
    with gr.Column(variant="panel"):
        table = gr.Dataframe(
            type="pandas",
            col_count=(100, "fixed"),
            label="Loan Application Attributes",
            datatype="number",
            height=450,
        )
        threshold = gr.Slider(
            0, 1, 0.5, 0.01, label="Potential Default Threshold"
        )
    upload_button.upload(upload_csv, upload_button, table)
    gr.Markdown("## Predictions")
    with gr.Row(variant="panel"):
        probability = gr.Number(
            label="Default Probability", precision=2, interactive=False
        )
        decision = gr.Textbox(label="Suggested Decision", interactive=False)
    table.select(predict, [table, threshold], [probability, decision])


app = gr.mount_gradio_app(app, demo, path="/app")
