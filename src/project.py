from src.chart_work import save_charts
from src.config import EDA_PATH, RAW_DATA_PATH, SCORES_PATH, SIMPLE_DATA_PATH, ensure_folders
from src.data_work import load_data, missing_table, notes_table, prepare_data, summary_numbers
from src.model_work import predict
from src.model_work import train_models as run_training


def build_project(data_path=RAW_DATA_PATH, force: bool = False):
    ensure_folders()
    raw_data = load_data(data_path)
    processed_data = prepare_data(raw_data)
    missing_data = missing_table(raw_data)
    notes = notes_table(raw_data)
    summary = summary_numbers(raw_data, processed_data)
    scores, best_model, importance = run_training(processed_data)
    charts = save_charts(raw_data, processed_data, scores, importance)

    processed_data.to_csv(SIMPLE_DATA_PATH, index=False)
    notes.to_csv(EDA_PATH, index=False)
    scores.to_csv(SCORES_PATH, index=False)

    return {
        "raw_data": raw_data,
        "processed_data": processed_data,
        "missing_data": missing_data,
        "notes": notes,
        "summary": summary,
        "scores": scores,
        "best_model": best_model,
        "importance": importance,
        "charts": charts,
        "force": force,
    }


def predict_survival(values: dict[str, float | int]):
    return predict(values)
