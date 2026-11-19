"""Load the two separately produced AI ratings of C/D answers."""

import math

import pandas as pd
from sklearn.metrics import cohen_kappa_score

import lab


MODE = "independent_ai_review"
FORM_NAMES = ("independent_review_1.json", "independent_review_2.json")
REPEAT_NAMES = ("independent_review_c_repeat_1.json", "independent_review_c_repeat_2.json")


def available():
    present = [(lab.DATA / name).is_file() for name in FORM_NAMES]
    if any(present) and not all(present):
        raise ValueError("Для независимой оценки нужен второй заполненный бланк.")
    return all(present)


def read_scores():
    template = pd.read_csv(lab.DATA / "reviews_blank_template.csv")
    samples = template.loc[template.type.isin(["C", "D"]), ["sample_id", "type"]]
    expected = dict(zip(samples.sample_id, samples.type))
    if samples.sample_id.duplicated().any() or samples.type.value_counts().to_dict() != {"C": 20, "D": 20}:
        raise ValueError("Ожидаются по 20 уникальных анонимных ответов C и D.")
    forms = []
    for name in FORM_NAMES:
        rows = lab.read_json(lab.DATA / name)
        if len(rows) != len(expected) or {row["sample_id"] for row in rows} != set(expected):
            raise ValueError(f"{name}: неверный набор sample_id.")
        for row in rows:
            kind = expected[row["sample_id"]]
            if row["type"] != kind or row["score"] not in (0, 1, 2) or not row["notes"].strip():
                raise ValueError(f"{name}: неверный тип, балл или комментарий.")
            if kind == "D":
                passed, result, error = (row[k] for k in ("pass1", "result_ok", "methodology_error"))
                valid = ((passed == 0 and result == 0 and error is None)
                         or (passed == 1 and result in (0, 1) and error in (0, 1)))
                if not valid or row["score"] != (0 if not passed else 2 if result and not error else 1):
                    raise ValueError(f"{name}: противоречивые поля для D.")
        forms.append(rows)

    def summarize(kind, rounds):
        scores = [{row["sample_id"]: row["score"] for row in rows if row["type"] == kind}
                  for rows in rounds]
        ids = sorted(scores[0])
        first, second = ([form[i] for i in ids] for form in scores)
        kappa = (float(cohen_kappa_score(first, second))
                 if len(set(first) | set(second)) > 1 else math.nan)
        return {"kappa": kappa if math.isfinite(kappa) else None,
                "agreement": sum(a == b for a, b in zip(first, second)) / len(ids),
                "n": len(ids), "evaluation_mode": MODE}

    first_c = summarize("C", forms)
    if first_c["kappa"] is None or first_c["kappa"] < 0.60:
        c_ids = {sample for sample, kind in expected.items() if kind == "C"}
        for i, name in enumerate(REPEAT_NAMES):
            if not (lab.DATA / name).is_file():
                raise ValueError("Первый раунд C требует двух повторных оценок.")
            rows = lab.read_json(lab.DATA / name)
            if (len(rows) != len(c_ids) or {row["sample_id"] for row in rows} != c_ids
                    or any(row["type"] != "C" or row["score"] not in (0, 1, 2)
                           or not row["notes"].strip() for row in rows)):
                raise ValueError(f"{name}: неверный повторный бланк C.")
            forms[i] = [row for row in forms[i] if row["type"] == "D"] + rows

    agreement = {kind: summarize(kind, forms) for kind in ("C", "D")}
    agreement["C"]["first_kappa"] = first_c["kappa"]
    agreement["C"]["first_agreement"] = first_c["agreement"]

    key = pd.read_csv(lab.DATA / "reviews_key.csv")
    merged = samples.merge(key, on="sample_id", validate="one_to_one")
    if len(merged) != len(samples):
        raise ValueError("Ключ не покрывает все анонимные ответы C/D.")
    a, b = (pd.DataFrame(rows).drop(columns="type") for rows in forms)
    merged = merged.merge(a, on="sample_id", validate="one_to_one").merge(
        b, on="sample_id", validate="one_to_one", suffixes=("_1", "_2"))

    full_1, full_2 = merged.score_1.eq(2), merged.score_2.eq(2)
    merged["manual_correct"] = full_1 & full_2
    merged["rater_disagreement"] = full_1.ne(full_2)
    merged["evaluation_mode"] = MODE
    merged["pass1"] = merged[["pass1_1", "pass1_2"]].min(axis=1, skipna=False)
    merged["result_ok"] = merged[["result_ok_1", "result_ok_2"]].min(axis=1, skipna=False)
    merged["methodology_error"] = merged[["methodology_error_1", "methodology_error_2"]].max(
        axis=1, skipna=False)
    columns = ["model", "task_id", "manual_correct", "score_1", "score_2", "rater_disagreement",
               "evaluation_mode", "pass1", "result_ok", "methodology_error"]
    return merged[columns], agreement
