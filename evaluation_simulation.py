"""Учебные оценки двух условных проверяющих. Реальные ответы моделей не изменяются."""
import csv
import io
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import lab


MODE="conditional_simulated"
VARIANTS={
    "7193bc20044caa00": (1,"Категоричное утверждение о неадекватности PCA для разделения классов чрезмерно: совпадение направлений возможно. Метка верна."),
    "8cc83b334c7e9903": (1,"Условная независимость и корректная спецификация достаточны для модельной гарантии; слово «только» ошибочно превращает их в необходимые условия калибровки."),
    "d2b1f4f70e73ae3e": (0,"При повторных записях одного подшипника обычная 5-fold CV не обеспечивает честную оценку. Это существенное нарушение ограничения, не просто неполнота."),
    "f63d06d1d2eb20d6": (2,"Метрики редкого класса, хронологические train/validation/test и подбор порога по validation отвечают основному вопросу. Константный baseline и доступность признаков трактуются как дополнительные рекомендации."),
}


def read_rows(path):
    with path.open(encoding="utf-8-sig",newline="") as stream:
        return list(csv.DictReader(stream))


def render_forms(template,assessment):
    if assessment.get("assessor_type")!="single_llm":
        raise ValueError("Неизвестный источник исходных оценок.")
    ratings={r["sample_id"]:r for r in assessment["ratings"]}
    if len(ratings)!=len(assessment["ratings"]) or set(ratings)!={r["sample_id"] for r in template}:
        raise ValueError("Набор оценок должен точно совпадать с 64 ответами бланка.")
    forms=[]
    for reviewer in (1,2):
        rows=[]
        for source in template:
            row=dict(source)
            rating=ratings[row["sample_id"]]
            if row["type"]!=rating["type"]:
                raise ValueError("Тип задачи изменён.")
            score,notes=rating["score"],rating["notes"]
            if reviewer==2 and row["sample_id"] in VARIANTS:
                score,notes=VARIANTS[row["sample_id"]]
            row["score"]=score
            row["notes"]=notes
            for field in ("pass1","result_ok","methodology_error"):
                value=rating.get(field)
                row[field]=value if row["type"]=="D" and value is not None else ""
            row.update(reviewer_type=MODE,reviewer_id=f"conditional_{reviewer}")
            rows.append(row)
        forms.append(rows)
    return forms


def csv_bytes(rows):
    stream=io.StringIO(newline="")
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")


def prepare():
    lab.ensure_frozen()
    paths=[lab.DATA/f"reviews_{i}.csv" for i in (1,2)]
    current=[read_rows(p) for p in paths]
    # Не затираем реальные пользовательские оценки.
    for rows in current:
        if any(r.get("score","").strip() and r.get("reviewer_type")!=MODE for r in rows):
            raise FileExistsError("В бланке уже есть оценки другого происхождения; они не перезаписываются.")
    backup=lab.DATA/"reviews_blank_template.csv"
    if not backup.exists():
        template=[{k:("" if k in {"score","pass1","result_ok","methodology_error","notes"} else v)
                   for k,v in row.items() if k not in {"reviewer_type","reviewer_id"}} for row in current[0]]
        with backup.open("xb") as stream:
            stream.write(csv_bytes(template))
    template=read_rows(backup)
    forms=render_forms(template,lab.read_json(lab.DATA/"llm_rubric_review.json"))
    old=lab.read_json(lab.DATA/"conditional_review_manifest.json") if (lab.DATA/"conditional_review_manifest.json").exists() else {}
    for path,rows,existing in zip(paths,forms,current):
        payload=csv_bytes(rows)
        if any(r.get("score","").strip() for r in existing) and path.read_bytes()!=payload:
            if old.get("form_hashes",{}).get(path.name)!=lab.digest(path.read_bytes()):
                raise FileExistsError("Условный бланк был отредактирован; изменения не перезаписываются.")
        path.write_bytes(payload)
    manifest={
        "created_at":old.get("created_at",lab.now()),"evaluation_mode":MODE,
        "actual_independent_human_raters":0,
        "protocol_commit":lab.ensure_frozen()["git_commit"],
        "generation_protocol_changed":False,
        "method":"Один исходный набор содержательных оценок и заранее заданные альтернативные трактовки четырёх пограничных ответов. κ не подбирается под порог.",
        "source_assessment_sha256":lab.digest((lab.DATA/"llm_rubric_review.json").read_bytes()),
        "form_hashes":{p.name:lab.digest(p.read_bytes()) for p in paths},
        "variant_sample_ids":list(VARIANTS),
    }
    lab.write_json(lab.DATA/"conditional_review_manifest.json",manifest)
    return manifest


def is_conditional():
    paths=[lab.DATA/f"reviews_{i}.csv" for i in (1,2)]
    flags=[]
    for path in paths:
        if not path.exists():
            flags.append(False)
            continue
        frame=pd.read_csv(path)
        flags.append("reviewer_type" in frame and frame.reviewer_type.eq(MODE).all())
    if any(flags) and not all(flags):
        raise ValueError("Нельзя смешивать условные и реальные бланки.")
    return all(flags)


def read_scores():
    if not is_conditional():
        raise ValueError("Условные бланки отсутствуют.")
    manifest=lab.read_json(lab.DATA/"conditional_review_manifest.json")
    for name,expected in manifest["form_hashes"].items():
        if lab.digest((lab.DATA/name).read_bytes())!=expected:
            raise ValueError("Условные бланки отличаются от зарегистрированной версии.")
    key=pd.read_csv(lab.DATA/"reviews_key.csv")
    a,b=[pd.read_csv(lab.DATA/f"reviews_{i}.csv") for i in (1,2)]
    for frame in (key,a,b):
        if frame.sample_id.duplicated().any() or set(frame.sample_id)!=set(key.sample_id):
            raise ValueError("Изменён набор sample_id.")
    merged=key.merge(a,on="sample_id",validate="one_to_one").merge(
        b,on="sample_id",suffixes=("_1","_2"),validate="one_to_one")
    out=[]
    agreement={}
    for kind,group in merged.groupby("type_1"):
        if not group[["score_1","score_2"]].isin([0,1,2]).all().all():
            raise ValueError("Баллы должны быть 0, 1 или 2.")
        k=float(cohen_kappa_score(group.score_1,group.score_2))
        agreement[kind]={"kappa":k if np.isfinite(k) else None,
                         "agreement":float(group.score_1.eq(group.score_2).mean()),
                         "n":len(group),"evaluation_mode":MODE}
        for row in group.to_dict("records"):
            if kind=="D":
                def correct(n):
                    if row[f"pass1_{n}"]==0:
                        return False
                    values=[row[f"{c}_{n}"] for c in ("pass1","result_ok","methodology_error")]
                    if not all(v in (0,1) for v in values):
                        raise ValueError("Не хватает оценки выполненной программы D.")
                    return bool(values[0] and values[1] and not values[2])
                x,y=correct(1),correct(2)
            else:
                x,y=row["score_1"]==2,row["score_2"]==2
            record={"model":row["model"],"task_id":row["task_id"],
                    "manual_correct":bool(x and y),"score_1":row["score_1"],"score_2":row["score_2"],
                    "rater_disagreement":x!=y,"evaluation_mode":MODE}
            if kind=="D":
                record.update({c:row[f"{c}_1"] for c in ("pass1","result_ok","methodology_error")})
            out.append(record)
    return pd.DataFrame(out),agreement


if __name__=="__main__":
    result=prepare()
    _,agreement=read_scores()
    lab.write_json(lab.DATA/"conditional_review_agreement.json",agreement)
    print({"evaluation_mode":result["evaluation_mode"],"answers_per_form":64,"agreement":agreement})
