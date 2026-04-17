from evaluate import load as load_metric

def compute_metrics(results):
    predictions = [r["prediction"] for r in results]
    references = [r["references"] for r in results]

    bleu = load_metric("bleu")
    rouge = load_metric("rouge")
    meteor = load_metric("meteor")

    return {
        "bleu1": bleu.compute(predictions=predictions, references=references, max_order=1)["bleu"],
        "bleu2": bleu.compute(predictions=predictions, references=references, max_order=2)["bleu"],
        "rougeL": rouge.compute(predictions=predictions, references=references)["rougeL"],
        "meteor": meteor.compute(predictions=predictions, references=references)["meteor"],
    }