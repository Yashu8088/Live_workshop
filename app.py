import gradio as gr
import pickle
import numpy as np
import pandas as pd

# 1. Load trained models
with open("cart_model.pkl", "rb") as f:
    cart_model = pickle.load(f)

with open("id3_model.pkl", "rb") as f:
    id3_model = pickle.load(f)

CLASS_NAMES = ["Setosa", "Versicolor", "Virginica"]

# Accept either canonical names or sklearn's original iris names
FEATURES_CANONICAL = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
FEATURES_SKLEARN = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]


def _get_model(model_type: str):
    """Helper to choose CART or ID3."""
    if model_type == "CART (Gini)":
        return cart_model
    return id3_model


# 2A. Single-row prediction
def predict_single(sepal_length, sepal_width, petal_length, petal_width, model_type):
    X = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    model = _get_model(model_type)

    probs = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(probs))

    return {
        "Chosen model": model_type,
        "Predicted class": CLASS_NAMES[pred_idx],
        "Probabilities": {
            "Setosa": float(probs[0]),
            "Versicolor": float(probs[1]),
            "Virginica": float(probs[2]),
        },
    }


# 2B. Batch prediction from uploaded CSV
def predict_batch(file, model_type):
    """
    file: uploaded CSV file from Gradio
    model_type: CART or ID3
    """
    if file is None:
        return pd.DataFrame({"error": ["Please upload a CSV file."]})

    # Try to read the CSV
    try:
        df = pd.read_csv(file.name)
    except Exception as e:
        return pd.DataFrame({"error": [f"Could not read CSV: {e}"]})

    # Handle sklearn-style column names by renaming to canonical
    cols = list(df.columns)

    if all(col in cols for col in FEATURES_SKLEARN):
        rename_map = {
            "sepal length (cm)": "sepal_length",
            "sepal width (cm)": "sepal_width",
            "petal length (cm)": "petal_length",
            "petal width (cm)": "petal_width",
        }
        df = df.rename(columns=rename_map)

    # Now check that canonical feature names exist
    if not all(col in df.columns for col in FEATURES_CANONICAL):
        return pd.DataFrame({
            "error": [
                "Input CSV must contain either:\n"
                "  - 'sepal_length','sepal_width','petal_length','petal_width'\n"
                "    OR\n"
                "  - 'sepal length (cm)','sepal width (cm)',"
                "'petal length (cm)','petal width (cm)'"
            ]
        })

    # Drop completely empty rows
    df = df.dropna(how="all")
    if df.empty:
        return pd.DataFrame({"error": ["All rows are empty after dropping NA."]})

    # Ensure numeric
    try:
        X = df[FEATURES_CANONICAL].astype(float).to_numpy()
    except Exception:
        return pd.DataFrame({
            "error": [
                "Feature columns must be numeric: "
                + ", ".join(FEATURES_CANONICAL)
            ]
        })

    model = _get_model(model_type)
    probs = model.predict_proba(X)
    preds = np.argmax(probs, axis=1)
    pred_labels = [CLASS_NAMES[i] for i in preds]

    # Build result DataFrame: original columns + predictions
    result = df.copy()
    result["predicted_class"] = pred_labels
    result["prob_setosa"] = probs[:, 0]
    result["prob_versicolor"] = probs[:, 1]
    result["prob_virginica"] = probs[:, 2]

    return result

def predict_batch_and_save(file, model_type):
    result = predict_batch(file, model_type)
    if not isinstance(result, pd.DataFrame):
        result = pd.DataFrame({"error": ["Unknown error"]})
    csv_path = "batch_predictions.csv"
    result.to_csv(csv_path, index=False)
    return csv_path


# 3. Gradio UI with Tabs
with gr.Blocks() as demo:
    gr.Markdown("# Decision Tree Classifier (CART vs ID3)")
    gr.Markdown(
        "Use the single prediction tab for one Iris flower, "
        "or upload a CSV file with multiple rows for batch prediction.\n\n"
        "**Data feeding happens entirely at the user end:** "
        "they prepare their own CSV, upload it, and see model outputs."
    )

    # ---- Tab 1: Single prediction ----
    with gr.Tab("Single prediction"):
        with gr.Row():
            sepal_length = gr.Number(label="Sepal length (cm)", value=5.1)
            sepal_width = gr.Number(label="Sepal width (cm)", value=3.5)
        with gr.Row():
            petal_length = gr.Number(label="Petal length (cm)", value=1.4)
            petal_width = gr.Number(label="Petal width (cm)", value=0.2)

        model_single = gr.Radio(
            choices=["CART (Gini)", "ID3 (Entropy)"],
            value="CART (Gini)",
            label="Decision tree type",
        )

        btn_single = gr.Button("Predict")
        out_single = gr.JSON(label="Prediction details")

        btn_single.click(
            fn=predict_single,
            inputs=[sepal_length, sepal_width, petal_length, petal_width, model_single],
            outputs=out_single,
        )

    # ---- Tab 2: Batch prediction (CSV upload) ----
    with gr.Tab("Batch prediction (CSV upload)"):
        gr.Markdown(
            "Upload a CSV file with column names either:\n"
            "- `sepal_length, sepal_width, petal_length, petal_width`, or\n"
            "- `sepal length (cm), sepal width (cm), petal length (cm), petal width (cm)`.\n\n"
            "You can edit your data in Excel / Python, save as CSV, upload here, "
            "and see the predictions instantly."
        )

        file_input = gr.File(label="Upload CSV file", file_types=[".csv"])
        model_batch = gr.Radio(
            choices=["CART (Gini)", "ID3 (Entropy)"],
            value="CART (Gini)",
            label="Decision tree type",
        )

        btn_batch = gr.Button("Run batch prediction")
        out_batch = gr.Dataframe(
            label="Predictions (input + model outputs)",
            interactive=False,
        )


        download_btn = gr.DownloadButton(
            label="Download results as CSV"
            # file_name="batch_predictions.csv"
        )

        # Show table
        btn_batch.click(
            fn=predict_batch,
            inputs=[file_input, model_batch],
            outputs=out_batch,
        )

        # Download CSV
        download_btn.click(
            fn=predict_batch_and_save,
            inputs=[file_input, model_batch],
            outputs=download_btn,
        )


       


# 4. Entry point
if __name__ == "__main__":
    demo.launch()