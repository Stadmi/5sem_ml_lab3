"""Представление сохранённых результатов для ipynb: без API, записи данных и запуска кода D.

Генерация и предзарегистрированная оценка остаются в неизменённом lab.py.
Этот модуль только собирает таблицы/рисунки для чтения и защиты работы.
"""
import re
import json
import subprocess
from datetime import datetime,timezone

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
import lab
import evaluation_simulation
import independent_review


TYPES=["A","B","C","D","F"]
TRUNCATED={"length","max_tokens","ALTERNATIVE_STATUS_TRUNCATED"}
MODEL_COLORS=("#1f77b4","#ff7f0e","#2ca02c","#d62728")


def model_names():
    return {m["name"]:(
        "Alice AI LLM" if m["name"]=="yandex" else
        "Qwen3 4B · локально" if m["name"]=="local" else
        "GPT-OSS 120B · Groq" if m["name"]=="foreign" else m["model"])
        for m in lab.config()["models"]}


def chart_label(name):
    return name.replace(" · Groq","").replace("Qwen3 4B · локально","Qwen3-4B")


def chart_colors():
    return dict(zip(model_names().values(),MODEL_COLORS))


def apply_c_final_decisions(scores, tasks, path):
    """Use the submitted C decisions for accuracy, alongside the original rubric scores."""
    decisions = pd.read_csv(path, dtype=str, keep_default_na=False)
    keys = ["model", "task_id"]
    if list(decisions.columns) != [*keys, "final_correct"]:
        raise ValueError("Неверные столбцы в файле итоговых зачётов C.")
    expected = pd.MultiIndex.from_product(
        [model_names(), tasks.loc[tasks.type.eq("C"), "id"]], names=keys)
    provided = decisions.set_index(keys)["final_correct"]
    if (not provided.index.is_unique or len(expected) != len(provided)
            or not expected.difference(provided.index).empty
            or not provided.isin(["True", "False"]).all()):
        raise ValueError("Итоговые зачёты C должны покрывать каждую пару модели и задачи одним True или False.")
    rows = scores.type.eq("C")
    index = pd.MultiIndex.from_frame(scores.loc[rows, keys])
    if not index.is_unique or len(index) != len(expected) or not expected.difference(index).empty:
        raise ValueError("Журнал ответов не покрывает все пары модели и задачи C.")
    scores.loc[rows, "correct"] = provided.reindex(index).eq("True").to_numpy()


def _bank_scores(bank_responses, tasks):
    lookup = {task["id"]: task for task in tasks.to_dict("records")}
    rows = []
    for response in bank_responses:
        task = lookup[response["task_id"]]
        parsed = lab.parse_answer(response["response"], response.get("finish_reason"))
        value = parsed["value"]
        usage = response["usage"]
        abstained = isinstance(value, str) and (
            value.upper() == "INSUFFICIENT_DATA"
            or any(term in value.lower() for term in
                   ("недостаточно данных", "данных недостаточно", "не могу", "неизвестно"))
        )
        rows.append({"model": response["model"], "task_id": task["id"], "type": task["type"],
                     "cluster": task["cluster"], "correct": lab.correct(task, value), **parsed,
                     "elapsed_s": response["elapsed_s"], "prompt_tokens": usage.get("prompt_tokens"),
                     "completion_tokens": usage.get("completion_tokens"), "abstained": abstained})
    return pd.DataFrame(rows)


def load_results():
    """Не читает секреты в таблицы и не перезаписывает результаты ручной проверки."""
    tasks=pd.DataFrame(lab.bank())
    local_log=lab.DATA/"responses.jsonl"
    log_path=local_log if local_log.is_file() else lab.DATA/"analysis_responses.jsonl"
    logs=lab.records(log_path)
    measured=[r for r in logs if r.get("phase") in {"bank","E"} and r.get("status")=="ok"]
    frame=pd.DataFrame(measured)
    if not frame.empty and frame.duplicated(["phase","model","task_id"]).any():
        raise ValueError("Повторные полученные ответы: нельзя произвольно выбрать лучший.")
    protocol=lab.ensure_frozen() if (lab.DATA/"preregistration.lock.json").exists() else None
    if protocol and any(r.get("protocol_commit")!=protocol["git_commit"] for r in measured):
        raise ValueError("Журнал содержит другой протокол; анализ версий нужно разделить.")
    scores=(lab.automatic_scores() if log_path==local_log else
            _bank_scores((r for r in measured if r["phase"]=="bank"),tasks))
    manual_note=""
    mode="preregistered"
    try:
        if evaluation_simulation.is_conditional():
            manual,agreement=evaluation_simulation.read_scores()
            mode=evaluation_simulation.MODE
        else:
            manual,agreement=lab.manual_scores()
        if independent_review.available():
            updated,updated_agreement=independent_review.read_scores()
            # The independent review covers C/D. B's explanations are legacy,
            # supplementary ratings; B accuracy still comes from the verdict.
            if not manual.empty:
                supplementary=manual[manual.task_id.str.startswith("B")]
                manual=pd.concat([supplementary,updated],ignore_index=True)
            else:
                manual=updated
            agreement={**{k:v for k,v in agreement.items() if k=="B"},**updated_agreement}
            mode=independent_review.MODE
    except (RuntimeError,ValueError) as exc:
        manual,agreement=pd.DataFrame(),{}
        manual_note=str(exc)
    if not scores.empty and not manual.empty:
        scores=scores.merge(manual,on=["model","task_id"],how="left",validate="one_to_one")
        take=scores.type.isin(["C","D"])
        scores.loc[take,"correct"]=scores.loc[take,"manual_correct"]
    c_decisions_path = lab.DATA / "c_final_decisions.csv"
    if c_decisions_path.is_file():
        if mode != independent_review.MODE or manual.empty:
            raise ValueError("Для итоговых зачётов C нужны исходные бланки независимой проверки.")
        apply_c_final_decisions(scores, tasks, c_decisions_path)
    return {"tasks":tasks,"logs":logs,"responses":frame,"scores":scores,"manual":manual,
            "agreement":agreement,"manual_note":manual_note,"protocol":protocol,"evaluation_mode":mode}


def coverage(bundle):
    expected_bank=int((bundle["tasks"].type!="E").sum())
    split_path=lab.DATA/"e_split.json"
    ntest=len(lab.read_json(split_path)["test"]) if split_path.exists() else 100
    rows=[]
    for model,name in model_names().items():
        for phase,expected in [("bank",expected_bank),("E",ntest*3)]:
            received=[r for r in bundle["logs"] if r.get("model")==model and
                      r.get("phase")==phase and r.get("status")=="ok"]
            rows.append({"модель":name,"часть":"Банк A/B/C/D/F" if phase=="bank" else "E: 0/5/20",
                         "получено":len(received),"ожидается":expected,
                         "обрывов":sum(r.get("finish_reason") in TRUNCATED for r in received),
                         "статус":"ответы получены" if len(received)==expected else "не завершено"})
    return pd.DataFrame(rows)


def task_matrix(bundle):
    ids=bundle["tasks"].query("type != 'E'").id.tolist()
    models=list(model_names())
    scores=bundle["scores"]
    matrix=(scores.pivot(index="task_id",columns="model",values="correct") if not scores.empty
            else pd.DataFrame())
    return matrix.reindex(index=ids,columns=models).rename(columns=model_names())


def model_metadata(bundle):
    rows=[]
    for m in lab.config()["models"]:
        samples=[r for r in bundle["logs"] if r.get("status")=="ok" and
                 r.get("phase") in {"bank","E"} and r.get("model")==m["name"]]
        versions=sorted({str(r.get("model_version") or r.get("system_fingerprint"))
                         for r in samples if r.get("model_version") or r.get("system_fingerprint")})
        effective=sorted({re.sub(r"gpt://[^/]+/","gpt://{folder_id}/",r["effective_model"])
                          for r in samples if r.get("effective_model")})
        stamps=[r["timestamp"] for r in samples]
        rows.append({"модель":model_names()[m["name"]],"заданный ID":m["model"],
                     "ID в ответе":"; ".join(effective) or "нет ответов",
                     "версия / fingerprint":"; ".join(versions) or "API не сообщает",
                     "temperature":m["temperature"],
                     "первый запрос UTC":min(stamps) if stamps else None,
                     "последний запрос UTC":max(stamps) if stamps else None,
                     "ответов bank+E":len(samples)})
    return pd.DataFrame(rows)


def protocol_timeline(bundle):
    """Проверка реальных дат, без изменения коммитов или журнала запросов."""
    lock=bundle.get("protocol")
    if not lock:
        return pd.DataFrame()
    measured=[r for r in bundle["logs"] if r.get("phase") in {"bank","E"}]
    first=min((datetime.fromisoformat(r["timestamp"]) for r in measured),default=None)
    frozen=datetime.fromisoformat(lock["created_at"])
    try:
        result=subprocess.run(["git","show","-s","--format=%cI",lock["git_commit"]],
                              cwd=lab.ROOT,capture_output=True,text=True,timeout=10)
        committed=datetime.fromisoformat(result.stdout.strip()) if result.returncode==0 else None
    except (OSError,ValueError,subprocess.TimeoutExpired):
        committed=None
    if (committed and committed>frozen) or (first and frozen>first):
        raise ValueError("Хронология не подтверждает фиксацию прогноза до проверки.")
    fmt=lambda value: value.astimezone(timezone.utc).isoformat() if value else "нет локальных сведений"
    return pd.DataFrame([
        {"этап":"Коммит гипотез и предварительного прогноза","время UTC":fmt(committed),
         "источник":"Git "+lock["git_commit"][:7]},
        {"этап":"Фиксация хэшей протокола","время UTC":fmt(frozen),
         "источник":"data/preregistration.lock.json"},
        {"этап":"Первый оценочный запрос","время UTC":fmt(first),
         "источник":"data/responses.jsonl"},
    ])


def accuracy_table(bundle):
    """Никакого среднего по произвольному подмножеству вместо общей accuracy."""
    scores=bundle["scores"]
    rows=[]
    for model,name in model_names().items():
        for kind in ["Все 36",*TYPES]:
            expected=set(bundle["tasks"].loc[
                bundle["tasks"].type.ne("E") if kind=="Все 36" else bundle["tasks"].type.eq(kind),"id"])
            group=(scores[scores.model.eq(model) & scores.task_id.isin(expected)]
                   if not scores.empty else pd.DataFrame())
            evaluated=group[group.correct.notna()] if not group.empty else group
            complete=len(group)==len(expected) and len(evaluated)==len(expected)
            mean=lo=hi=clo=chi=np.nan
            if complete:
                values=evaluated.correct.astype(float)
                mean=float(values.mean())
                lo,hi=lab.bootstrap_ci(values)
                clo,chi=lab.bootstrap_ci(values,clusters=evaluated.cluster)
            rows.append({"модель":name,"тип":kind,"оценено":len(evaluated),"всего":len(expected),
                         "accuracy":mean,"ДИ 2.5%":lo,"ДИ 97.5%":hi,
                         "кластерный ДИ 2.5%":clo,"кластерный ДИ 97.5%":chi,
                         "статус":"готово" if complete else "ожидаются ответы / ручные оценки"})
    result=pd.DataFrame(rows)
    if bundle.get("evaluation_mode") in (evaluation_simulation.MODE,independent_review.MODE):
        result["оценивание C/D"]="Проверяющий 1 / Проверяющий 2"
    return result


def pairwise_table(bundle):
    accuracy=accuracy_table(bundle)
    if not accuracy.query("тип == 'Все 36'").статус.eq("готово").all():
        return pd.DataFrame()
    table=lab.compare_models(bundle["scores"]).copy()
    for key in ("model_a","model_b"):
        table[key]=table[key].map(model_names())
    return table


def response_quality(bundle):
    rows=[]
    for model,name in model_names().items():
        group=(bundle["scores"].query("model == @model") if not bundle["scores"].empty else pd.DataFrame())
        if group.empty:
            continue
        required=group.task_id.isin(["F07","F08"])
        samples=[r for r in bundle["logs"] if r.get("status")=="ok" and
                 r.get("phase")=="bank" and r.get("model")==model]
        rows.append({"модель":name,"ответов":len(group),
                     "отказов (текст)":int(group.abstained.sum()),
                     "отказов на F07/F08":int(group.loc[required,"abstained"].sum()),
                     "отказов на остальных":int(group.loc[~required,"abstained"].sum()),
                     "не распознано ОТВЕТ":int((~group.parse_ok).sum()),
                     "обрывов":sum(r.get("finish_reason") in TRUNCATED for r in samples),
                     "покрытие уверенности":float(group.confidence.notna().mean())})
    return pd.DataFrame(rows)


def review_progress():
    rows=[]
    for reviewer in (1,2):
        path=lab.DATA/f"reviews_{reviewer}.csv"
        if not path.exists():
            continue
        frame=pd.read_csv(path)
        for kind in "BCD":
            group=frame[frame.type.eq(kind)]
            done=group.score.isin([0,1,2])
            if kind=="D":
                valid=group[["pass1","result_ok","methodology_error"]].isin([0,1]).all(axis=1)
                if "reviewer_type" in group and group.reviewer_type.eq(evaluation_simulation.MODE).all():
                    valid |= group.pass1.eq(0) & group.result_ok.eq(0)
                done &= valid
            rows.append({"проверяющий":reviewer,"тип":kind,"полностью заполнено":int(done.sum()),
                         "нужно":len(group)})
    return pd.DataFrame(rows)


def code_observations():
    path=lab.DATA/"code_execution.json"
    if not path.exists():
        return pd.DataFrame()
    data=lab.read_json(path)
    fields=["sample_id","task_id","execution","pass1_observed","returncode","timed_out",
            "elapsed_s","reference_metric","last_stdout_number_candidate"]
    return pd.DataFrame(data["samples"]).reindex(columns=fields)


def code_metrics(bundle):
    manual=bundle.get("manual",pd.DataFrame())
    rows=[]
    for model,name in model_names().items():
        group=(manual[manual.model.eq(model) & manual.task_id.str.startswith("D")]
               if not manual.empty else pd.DataFrame())
        expected=int(bundle["tasks"].type.eq("D").sum())
        complete=len(group)==expected and expected>0
        fully_correct=(group.manual_correct if "manual_correct" in group else
                       group.pass1.eq(1) & group.result_ok.eq(1) & group.methodology_error.eq(0)) if complete else None
        rows.append({"модель":name,"проверено":len(group),"нужно":expected,
                     "pass@1":float(group.pass1.mean()) if complete else np.nan,
                     "полная корректность D":float(fully_correct.mean()) if complete else np.nan,
                     "результат в допуске":float(group.result_ok.mean()) if complete else np.nan,
                     "методологических ошибок":float(group.methodology_error.mean()) if complete else np.nan,
                     "статус":"готово" if complete else "нет полной независимой проверки"})
    return pd.DataFrame(rows)


def supplementary_review(bundle):
    """Один явно обозначенный ИИ-оценщик. Никогда не подмешивается в human/manual_scores."""
    path=lab.DATA/"llm_rubric_review.json"
    if not path.exists():
        return pd.DataFrame()
    data=lab.read_json(path)
    if data.get("assessor_type")!="single_llm" or data.get("blind") is not False or data.get("independent_human_raters")!=0:
        raise ValueError("Дополнительная проверка должна явно раскрывать одного неслепого ИИ-оценщика.")
    tasks=bundle["tasks"].set_index("id")
    expected={lab.digest(row["request_id"])[:16]:row for row in bundle["logs"]
              if row.get("phase")=="bank" and row.get("status")=="ok"
              and tasks.loc[row["task_id"],"type"] in "BCD"}
    ratings=data["ratings"]
    ids=[r["sample_id"] for r in ratings]
    if len(ids)!=len(set(ids)) or set(ids)!=set(expected):
        raise ValueError("Дополнительная проверка должна покрывать ровно полученные B/C/D без дубликатов.")
    code_path=lab.DATA/"code_execution.json"
    observed={r["sample_id"]:r for r in lab.read_json(code_path)["samples"]} if code_path.exists() else {}
    rows=[]
    for rating in ratings:
        row=expected[rating["sample_id"]]
        if rating["score"] not in (0,1,2) or rating["type"]!=tasks.loc[row["task_id"],"type"]:
            raise ValueError("Тип или балл дополнительной проверки неверен.")
        if not rating.get("notes"):
            raise ValueError("Нужно содержательное обоснование балла.")
        if rating.get("quote") and rating["quote"] not in row["response"]:
            raise ValueError("Цитата ИИ-оценки отсутствует в исходном ответе: "+rating["sample_id"])
        if rating["type"]=="D":
            execution=observed.get(rating["sample_id"])
            if not execution or rating["pass1"]!=execution["pass1_observed"]:
                raise ValueError("Оценка D не подтверждается реальным первым запуском.")
            if execution["execution"]=="not_executable":
                if row.get("finish_reason") not in TRUNCATED or rating.get("finish_reason")!=row.get("finish_reason"):
                    raise ValueError("Отсутствие кода должно подтверждаться сохранённым обрывом.")
            else:
                if not rating.get("quote"):
                    raise ValueError("Для готового кода нужна цитата.")
                if rating["result_ok"]:
                    value=execution.get("last_stdout_number_candidate")
                    if value is None or abs(value-execution["reference_metric"])>execution["metric_atol"]:
                        raise ValueError("Заявленный результат D не подтверждается stdout.")
            correct=bool(rating["pass1"] and rating["result_ok"] and rating["methodology_error"]==0)
        else:
            if not rating.get("quote"):
                raise ValueError("Для содержательной оценки нужна цитата.")
            correct=rating["score"]==2
        rows.append({**rating,"request_id":row["request_id"],"model":row["model"],"task_id":row["task_id"],
                     "correct_single_llm":correct,"response_sha256":lab.digest(row["response"].encode("utf-8")),
                     "basis":"один ИИ, неслепая оценка; не две человеческие проверки"})
    return pd.DataFrame(rows)


def supplementary_summary(bundle,review):
    """Отдельный разведочный итог, не официальная accuracy по предрегистрации."""
    if review.empty:
        return pd.DataFrame()
    scores=bundle["scores"].copy()
    replacements=review[review.type.isin(["C","D"])].set_index(["model","task_id"]).correct_single_llm
    for index,row in scores[scores.type.isin(["C","D"])].iterrows():
        scores.at[index,"correct"]=replacements.loc[(row["model"],row["task_id"])]
    rows=[]
    for model,group in scores.groupby("model"):
        expected=set(bundle["tasks"].loc[bundle["tasks"].type.ne("E"),"id"])
        if set(group.task_id)!=expected or group.correct.isna().any():
            continue
        rows.append({"модель":model_names()[model],"задач":len(group),
                     "верных (с оценкой ИИ C/D)":int(group.correct.astype(int).sum()),
                     "accuracy (дополнительная)":float(group.correct.astype(float).mean()),
                     "основание":"A/B/F: авто; C/D: один ИИ, неслепая оценка",
                     "две человеческие оценки":False})
    return pd.DataFrame(rows)


def classification_predictions(bundle):
    path=lab.DATA/"e_split.json"
    if not path.exists():
        return pd.DataFrame()
    split=lab.read_json(path)
    y=(load_breast_cancer().target==0).astype(int)
    rows=[]
    for r in bundle["logs"]:
        if r.get("phase")!="E" or r.get("status")!="ok":
            continue
        _,shots,row_id=r["task_id"].split(":")
        row_id=int(row_id)
        if row_id not in split["test"]:
            raise ValueError("В E есть строка вне фиксированного test.")
        value=lab.parse_answer(r["response"],r.get("finish_reason"))["value"]
        pred=int(value) if type(value) in (int,float) and value in (0,1) else -1
        rows.append({"model":r["model"],"shots":int(shots),"row_id":row_id,"true":int(y[row_id]),
                     "pred":pred,"truncated":r.get("finish_reason") in TRUNCATED})
    return pd.DataFrame(rows)


def classification_table(bundle):
    pred=classification_predictions(bundle)
    split=lab.read_json(lab.DATA/"e_split.json") if (lab.DATA/"e_split.json").exists() else {"test":[]}
    rows=[]
    for model,name in model_names().items():
        for shots in (0,5,20):
            group=(pred[pred.model.eq(model) & pred.shots.eq(shots)] if not pred.empty else pred)
            complete=bool(split["test"]) and set(group.row_id if not group.empty else [])==set(split["test"])
            row={"модель":name,"примеров":shots,"получено":len(group),"ожидается":len(split["test"]),
                 "статус":"готово" if complete else "неполная серия — метрики не считаются",
                 "F1":np.nan,"balanced accuracy":np.nan,"accuracy":np.nan,"valid rate":np.nan,
                 "F1 ДИ снизу":np.nan,"F1 ДИ сверху":np.nan,
                 "BA ДИ снизу":np.nan,"BA ДИ сверху":np.nan,
                 "обрывов":int(group.truncated.sum()) if not group.empty else 0}
            if complete:
                group=group.set_index("row_id").loc[split["test"]].reset_index()
                metrics=lab.class_metrics(group.true,group.pred)
                ci=lab.classification_ci(group.true,group.pred)
                row.update({"F1":metrics["f1"],"balanced accuracy":metrics["balanced_accuracy"],
                            "accuracy":metrics["accuracy"],"valid rate":metrics["valid_rate"],
                            "F1 ДИ снизу":ci["f1_ci_low"],"F1 ДИ сверху":ci["f1_ci_high"],
                            "BA ДИ снизу":ci["balanced_accuracy_ci_low"],
                            "BA ДИ сверху":ci["balanced_accuracy_ci_high"]})
            rows.append(row)
    return pd.DataFrame(rows)


def calibration_tables(bundle):
    summaries=[]
    bins={}
    scores=bundle["scores"]
    if scores.empty:
        return pd.DataFrame(),bins
    for model,group in scores[scores.type.isin(["A","B","F"])].groupby("model"):
        valid=group[group.confidence.notna() & group.correct.notna()]
        if valid.empty:
            continue
        ece,binned=lab.calibration(valid.correct.astype(float),valid.confidence.astype(float))
        bins[model_names()[model]]=binned
        summaries.append({"модель":model_names()[model],"ответов с уверенностью":len(valid),
                          "всего A/B/F":len(group),"покрытие":len(valid)/len(group),
                          "средняя уверенность":valid.confidence.mean(),
                          "accuracy на том же подмножестве":valid.correct.astype(float).mean(),"ECE":ece})
    return pd.DataFrame(summaries),bins


def cost_latency(bundle):
    rows=[]
    accuracy=accuracy_table(bundle)
    for m in lab.config()["models"]:
        for phase in ("bank","E"):
            group=[r for r in bundle["logs"] if r.get("model")==m["name"] and
                   r.get("phase")==phase and r.get("status")=="ok"]
            if not group:
                continue
            price=lab.pricing_for(m)
            known=all(price.get(k) is not None for k in ("input_rub_per_million","output_rub_per_million"))
            known=known and bool(price.get("source") and price.get("accessed_at"))
            usage_ok=all(all(r["usage"].get(k) is not None for k in ("prompt_tokens","completion_tokens")) for r in group)
            cost=np.nan
            tokens=np.nan
            if usage_ok:
                ti=sum(r["usage"]["prompt_tokens"] for r in group)
                to=sum(r["usage"]["completion_tokens"] for r in group)
                tokens=(ti+to)/len(group)
                if known:
                    cost=(ti*price["input_rub_per_million"]+to*price["output_rub_per_million"])/1e6
            overall=accuracy[accuracy["модель"].eq(model_names()[m["name"]]) & accuracy.тип.eq("Все 36")].iloc[0]
            per_correct=np.nan
            if phase=="bank" and overall.статус=="готово" and overall.accuracy>0 and np.isfinite(cost):
                per_correct=cost/(overall.accuracy*overall["всего"])
            rows.append({"модель":model_names()[m["name"]],"часть":phase,"получено ответов":len(group),
                         "API задержка средняя, с":float(np.mean([r["elapsed_s"] for r in group])),
                         "токенов на ответ":tokens,"номинальная стоимость, ₽":cost,
                         "₽ за верный ответ банка":per_correct,
                         "тариф":"из tariffs.json" if known else "не подтверждён / локальная модель"})
    return pd.DataFrame(rows)


def hypothesis_results(bundle):
    """Read-only представление формул lab.hypotheses; совпадение проверяется тестом."""
    result={"H1":[],"H5":None,"H10":[]}
    scores=bundle["scores"]
    if scores.empty:
        return result
    automatic=scores[scores.type.isin(["A","B","F"]) & scores.correct.notna()].copy()
    a_ids=[a for a,f in lab.H1_PAIRS]
    f_ids=[f for a,f in lab.H1_PAIRS]
    expected_cal=set(bundle["tasks"].loc[bundle["tasks"].type.isin(["A","B","F"]),"id"])
    verdict=lambda ci: "поддержана" if ci[0]>0 else "противоречие" if ci[1]<0 else "данных недостаточно"
    for model,group in automatic.groupby("model"):
        values=group.set_index("task_id").correct
        if set(a_ids+f_ids)<=set(values.index):
            av,fv=values.loc[a_ids].to_numpy(dtype=float),values.loc[f_ids].to_numpy(dtype=float)
            table=[[int(((av==1)&(fv==1)).sum()),int(((av==1)&(fv==0)).sum())],
                   [int(((av==0)&(fv==1)).sum()),int(((av==0)&(fv==0)).sum())]]
            result["H1"].append({"model":model,"delta":float((av-fv).mean()),
                                 "accuracy_a":float(av.mean()),"accuracy_f":float(fv.mean()),
                                 "only_a":table[0][1],"only_f":table[1][0],
                                 "ci":lab.bootstrap_ci(av-fv),"p":float(mcnemar(table,exact=True).pvalue),
                                 "pairs":len(lab.H1_PAIRS)})
        cal=group[group.confidence.notna()]
        if len(cal) and set(group.task_id)==expected_cal:
            ece,_=lab.calibration(cal.correct.astype(float),cal.confidence.astype(float))
            delta=cal.confidence.astype(float)-cal.correct.astype(float)
            ci=lab.bootstrap_ci(delta,clusters=cal.cluster)
            result["H10"].append({"model":model,"delta":float(delta.mean()),"ci":ci,"ece":ece,
                                  "confidence_coverage":len(cal)/len(group),"status":verdict(ci),
                                  "exploratory":True})
    if result["H1"]:
        adjusted=multipletests([r["p"] for r in result["H1"]]+[1.]*(4-len(result["H1"])),method="holm")[1]
        for row,p in zip(result["H1"],adjusted):
            row["p_holm"]=float(p)
            row["status"]=("поддержана" if row["delta"]>0 else "противоречие") if p<.05 else "данных недостаточно"
    cfg=lab.config()["models"]
    local=next(m["name"] for m in cfg if m["role"]=="local_small")
    large=[m["name"] for m in cfg if m["role"]!="local_small"]
    pivot=automatic[automatic.type.isin(["A","B"])].pivot(index="task_id",columns="model",values="correct")
    ids_a=bundle["tasks"].loc[bundle["tasks"].type.eq("A"),"id"].tolist()
    ids_b=bundle["tasks"].loc[bundle["tasks"].type.eq("B"),"id"].tolist()
    if set([local,*large])<=set(pivot.columns) and set(ids_a+ids_b)<=set(pivot.index):
        pivot=pivot.loc[ids_a+ids_b,[local,*large]]
        if not pivot.isna().any().any():
            large_score=pivot[large].astype(float).mean(axis=1)
            local_score=pivot[local].astype(float)
            gap=large_score-local_score
            a,b=gap.loc[ids_a].to_numpy(),gap.loc[ids_b].to_numpy()
            rng=np.random.default_rng(lab.SEED)
            boot=a[rng.integers(0,len(a),(1000,len(a)))].mean(1)-b[rng.integers(0,len(b),(1000,len(b)))].mean(1)
            ci=np.percentile(boot,[2.5,97.5]).tolist()
            result["H5"]={"gap_A":float(a.mean()),"gap_B":float(b.mean()),
                          "large_A":float(large_score.loc[ids_a].mean()),"local_A":float(local_score.loc[ids_a].mean()),
                          "large_B":float(large_score.loc[ids_b].mean()),"local_B":float(local_score.loc[ids_b].mean()),
                          "difference":float(a.mean()-b.mean()),"ci":ci,"status":verdict(ci)}
    return result


def hypothesis_table(result,key):
    rows=result.get(key)
    if not rows:
        return pd.DataFrame()
    table=pd.DataFrame(rows if isinstance(rows,list) else [rows])
    if "model" in table:
        table["model"]=table.model.map(model_names())
    return table


def verified_errors(bundle):
    path=lab.DATA/"error_candidates.json"
    if not path.exists():
        return pd.DataFrame()
    logs={r["request_id"]:r for r in bundle["logs"] if r.get("status")=="ok"}
    errors=lab.read_json(path)["errors"]
    for err in errors:
        row=logs.get(err["request_id"])
        if not row or row["task_id"]!=err["task_id"] or row["model"]!=err["model"] or err["quote"] not in row["response"]:
            raise ValueError("Цитата разбора ошибки не подтверждена исходным журналом.")
    return pd.DataFrame(errors).assign(model=lambda f:f.model.map(model_names()))


def plot_accuracy(table):
    import matplotlib.pyplot as plt
    ready=table.query("тип == 'Все 36' and статус == 'готово'")
    fig,ax=plt.subplots(figsize=(8,3.5))
    if ready.empty:
        ax.text(.5,.5,"Общая точность пока не определена:\nнужны независимые оценки C/D.",
                ha="center",va="center",transform=ax.transAxes)
        ax.set_axis_off()
    else:
        colors=chart_colors()
        ranked=ready.set_index("модель")
        ready=ranked.loc[[name for name in colors if name in ranked.index]].reset_index()
        for i,row in enumerate(ready.to_dict("records")):
            ax.errorbar(row["accuracy"],i,xerr=[[row["accuracy"]-row["ДИ 2.5%"]],
                        [row["ДИ 97.5%"]-row["accuracy"]]],fmt="o",capsize=4,color=colors[row["модель"]])
            ax.annotate(f'{row["accuracy"]:.0%}',(row["ДИ 97.5%"],i),xytext=(4,0),
                        textcoords="offset points",va="center",color=colors[row["модель"]])
        ax.set(yticks=range(len(ready)),yticklabels=[chart_label(n) for n in ready["модель"]],
               xlim=(0,1.12),xlabel="Доля верных ответов на 36 задачах (95% ДИ, бутстреп)")
        ax.set_xticks(np.linspace(0,1,6))
        ax.invert_yaxis()
        ax.grid(alpha=.22)
    ax.set_title("Точность моделей с доверительными интервалами")
    fig.tight_layout()
    return fig


def heatmap_values(table,classification=None,rubric=None,classification_metric="balanced accuracy"):
    """Accuracy; при переданной рубрике C — средний балл / 2; E — BA."""
    pivot=table[table.тип.isin(TYPES)].pivot(index="модель",columns="тип",values="accuracy")
    pivot=pivot.reindex(index=list(model_names().values()),columns=TYPES)
    if rubric is not None:
        column="средний балл C (0–2)"
        if rubric["модель"].duplicated().any():
            raise ValueError("Повторная оценка C для одной модели.")
        values=pd.to_numeric(rubric[column],errors="raise")
        if (values.notna() & ~values.between(0,2)).any():
            raise ValueError("Средний балл C должен быть в диапазоне 0–2.")
        normalized=rubric.assign(**{column:values}).set_index("модель")[column]/2
        pivot["C"]=normalized.reindex(pivot.index)
    if classification is not None:
        ready=classification[classification["примеров"].eq(20) & classification["статус"].eq("готово")]
        if ready["модель"].duplicated().any():
            raise ValueError("Повторная итоговая метрика E для одной модели.")
        pivot["E"]=ready.set_index("модель")[classification_metric].reindex(pivot.index)
    return pivot


def plot_heatmap(table,classification=None,rubric=None,classification_metric="balanced accuracy"):
    import matplotlib.pyplot as plt
    pivot=heatmap_values(table,classification,rubric,classification_metric)
    fig,ax=plt.subplots(figsize=(9,4))
    cmap=plt.get_cmap("RdYlGn").with_extremes(bad="#e5e7eb")
    image=ax.imshow(pivot.to_numpy(dtype=float),vmin=0,vmax=1,cmap=cmap,aspect="auto")
    descriptions={"A":"A\nрасчёт","B":"B\nверно/неверно","C":"C\nвыбор метода",
                  "D":"D\nкод","F":"F\nловушки",
                  "E":"E\n"+("accuracy" if classification_metric=="accuracy" else "BA")+", 20 прим."}
    if rubric is not None:
        descriptions["C"]="C\nбалл / 2"
    labels=[descriptions.get(c,c) for c in pivot.columns]
    ax.set(xticks=range(len(pivot.columns)),xticklabels=labels,
           yticks=range(len(pivot)),yticklabels=[chart_label(n) for n in pivot.index],
           title="Качество по типам задач" if rubric is not None else "Доля верных ответов по типам задач")
    for i,j in np.ndindex(pivot.shape):
        value=pivot.iloc[i,j]
        red,green,blue,_=cmap(value if pd.notna(value) else np.nan)
        luminance=.2126*red+.7152*green+.0722*blue
        ax.text(j,i,"—" if pd.isna(value) else f"{value:.0%}",ha="center",va="center",
                color="white" if luminance<.48 else "black")
    fig.colorbar(image,ax=ax)
    fig.tight_layout()
    return fig


def plot_code_quality(metrics):
    """Первый успешный запуск и полное выполнение задания показаны раздельно."""
    import matplotlib.pyplot as plt
    ready=metrics.set_index("модель").reindex(list(model_names().values()))
    x=np.arange(len(ready))
    fig,ax=plt.subplots(figsize=(9,3.5))
    for shift,column,label,color in [(-.18,"pass@1","Запускаемость (pass@1)","#4c78a8"),
                                    (.18,"полная корректность D","Полная корректность D","#f58518")]:
        values=ready[column].astype(float)
        bars=ax.bar(x+shift,values,width=.34,label=label,color=color)
        ax.bar_label(bars,labels=["—" if pd.isna(v) else f"{v:.0%}" for v in values],padding=3)
    ax.set(xticks=x,xticklabels=["GigaChat-2-Max","Alice AI LLM","GPT-OSS 120B","Qwen3-4B"],
           ylim=(0,1.25),yticks=np.linspace(0,1,5),ylabel="Доля программ",title="Код D: запуск и выполнение задания")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value,pos:f"{value:.0%}"))
    ax.legend(loc="upper center",ncol=2,fontsize=9)
    fig.tight_layout()
    return fig


def reviewer_summary(bundle):
    """Согласованность и средние баллы двух сохранённых наборов оценок."""
    manual=bundle.get("manual",pd.DataFrame())
    rows=[]
    for kind,agreement in bundle["agreement"].items():
        group=manual[manual.task_id.str.startswith(kind)]
        rows.append({"тип":kind,"ответов":agreement["n"],
                     "Проверяющий 1":float(group.score_1.mean()),
                     "Проверяющий 2":float(group.score_2.mean()),
                     "доля совпадений":agreement["agreement"],"каппа Коэна":agreement["kappa"]})
    return pd.DataFrame(rows)


def rubric_summary(bundle):
    manual=bundle.get("manual",pd.DataFrame())
    rows=[]
    for model,name in model_names().items():
        row={"модель":name}
        for kind in ("B","C"):
            group=(manual[manual.model.eq(model) & manual.task_id.str.startswith(kind)]
                   if not manual.empty else pd.DataFrame())
            expected=int(bundle["tasks"].type.eq(kind).sum())
            row[f"средний балл {kind} (0–2)"]=(
                float(group[["score_1","score_2"]].to_numpy(dtype=float).mean())
                if expected and len(group)==expected else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def presentation_table(frame):
    """Не меняет числовые данные: только поясняет неприменимые/неизвестные значения."""
    result=frame.astype(object).copy()
    for index,row in frame.iterrows():
        for column in frame.columns:
            value=row[column]
            if isinstance(value,(list,tuple,dict)) or not pd.isna(value):
                continue
            reason="не определено"
            if column=="методологических ошибок" and row.get("pass@1")==0:
                reason="не оценивается: нет завершённого кода"
            elif row.get("execution")=="not_executable":
                reason="нет завершённого кода"
            elif column=="last_stdout_number_candidate":
                reason="метрика не напечатана"
            elif column=="₽ за верный ответ банка" and row.get("часть")=="E":
                reason="не применяется к типу E"
            elif "₽" in column:
                reason=("локальные затраты не оценены" if row.get("модель")==model_names()["local"]
                        else "тариф не подтверждён")
            result.at[index,column]=reason
    return result


def plot_classification(table,baselines):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),sharey=True)
    colors=chart_colors()
    for ax,col,basecol in zip(axes,["F1","balanced accuracy"],["f1","balanced_accuracy"]):
        for name,color in colors.items():
            group=table[table["модель"].eq(name)].sort_values("примеров")
            ax.plot(group["примеров"],group[col],"o-",label=chart_label(name),color=color)
        if not baselines.empty:
            for name,group in baselines.groupby("model"):
                if name=="Always frequent class":
                    ax.axhline(group[basecol].iloc[0],color="black",linestyle=":",label="Константа")
                else:
                    ax.plot(group.shots,group[basecol],"s--",label=name,alpha=.8,
                            color={"LogisticRegression":"#555555","GaussianNB":"#999999"}.get(name))
        ax.set(xticks=[0,5,20],xlabel="Число примеров в промпте / обучении",
               title="F1 класса malignant" if col=="F1" else "Balanced accuracy",ylim=(-.03,1.03))
        ax.grid(alpha=.2)
    handles,labels=axes[1].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=4,fontsize=8)
    fig.suptitle("Тип E: LLM и классические модели на одних и тех же примерах")
    fig.tight_layout(rect=(0,.13,1,1))
    return fig


def plot_calibration(bins,summary=None):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6,5))
    ax.plot([0,1],[0,1],"k--",linewidth=1,label="Идеальная калибровка")
    ece=summary.set_index("модель")["ECE"].to_dict() if summary is not None and not summary.empty else {}
    for name,color in chart_colors().items():
        if name not in bins:
            continue
        table=bins[name]
        valid=table[table.n.gt(0)]
        label=chart_label(name)+(f" (ECE={ece[name]:.3f})" if name in ece else "")
        ax.scatter(valid.confidence,valid.accuracy,label=label,color=color,s=45)
    ax.set(xlim=(0,1.04),ylim=(0,1.04),xlabel="Заявленная уверенность (среднее в группе)",
           ylabel="Доля верных ответов в группе",title="Диаграмма надёжности на A/B/F")
    ax.grid(alpha=.22)
    ax.legend(fontsize=8,loc="lower right")
    fig.tight_layout()
    return fig


def plot_hypothesis(result,key):
    import matplotlib.pyplot as plt
    table=hypothesis_table(result,key)
    fig,ax=plt.subplots(figsize=(8,4))
    colors=chart_colors()
    if table.empty:
        ax.text(.5,.5,"Нет полного набора данных для этой проверки.",
                ha="center",va="center",transform=ax.transAxes)
        ax.set_axis_off()
    elif key=="H1":
        ordered=table.set_index("model")
        ordered=ordered.loc[[name for name in colors if name in ordered.index]]
        x=np.arange(len(ordered))
        ax.bar(x-.19,ordered.accuracy_a,.38,color=MODEL_COLORS[0],label="Исходные задачи A")
        ax.bar(x+.19,ordered.accuracy_f,.38,color=MODEL_COLORS[1],label="Изменённые задачи F")
        ax.set(xticks=x,xticklabels=[chart_label(n) for n in ordered.index],ylim=(0,1.13),
               ylabel="Доля верных ответов",title=f"H1: исходные и изменённые задачи ({int(ordered.pairs.iloc[0])} пар)")
        ax.legend(loc="upper center",ncol=2,fontsize=8)
    elif key=="H5":
        row=table.iloc[0]
        x=np.arange(2)
        ax.bar(x-.19,[row.large_A,row.large_B],.38,color=MODEL_COLORS[2],label="Большие (среднее)")
        ax.bar(x+.19,[row.local_A,row.local_B],.38,color=MODEL_COLORS[3],label="Qwen3-4B")
        for i,kind in enumerate("AB"):
            ax.text(i,max(row[f"large_{kind}"],row[f"local_{kind}"])+.035,
                    f"Разрыв {row[f'gap_{kind}']:+.1%}",ha="center",fontsize=9)
        ax.set(xticks=x,xticklabels=["A (расчёт)","B (теория)"],ylim=(0,1.13),
               ylabel="Доля верных ответов",title="H5: отставание Qwen3-4B от больших моделей")
        ax.legend(loc="lower right",fontsize=8)
    elif key=="H10":
        ordered=table.set_index("model")
        ordered=ordered.loc[[name for name in colors if name in ordered.index]]
        for i,(name,row) in enumerate(ordered.iterrows()):
            lo,hi=row["ci"]
            value=row["delta"]
            ax.errorbar(i,value,yerr=[[max(0,value-lo)],[max(0,hi-value)]],
                        fmt="o",capsize=4,color=colors[name])
        ax.axhline(0,color="black",linewidth=1)
        ax.set(xticks=range(len(ordered)),xticklabels=[chart_label(n) for n in ordered.index],
               ylabel="Уверенность − точность (95% ДИ)",title="H10: положительный разрыв означает завышенную уверенность")
    ax.grid(axis="y",alpha=.22)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def plot_cost_accuracy(costs,accuracy):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4.4))
    ready=accuracy.query("тип == 'Все 36' and статус == 'готово'")
    bank=costs[costs["часть"].eq("bank")] if not costs.empty else costs
    points=(ready.merge(bank,on="модель") if not ready.empty and not bank.empty else pd.DataFrame())
    priced=points.dropna(subset=["₽ за верный ответ банка"]) if not points.empty else points
    if priced.empty:
        ax.text(.5,.5,"Нет подтверждённой денежной оценки",ha="center",transform=ax.transAxes)
    else:
        colors=chart_colors()
        for row in priced.to_dict("records"):
            name=row["модель"]
            ax.errorbar(row["₽ за верный ответ банка"],row["accuracy"],color=colors[name],
                        yerr=[[max(0,row["accuracy"]-row["ДИ 2.5%"])],
                              [max(0,row["ДИ 97.5%"]-row["accuracy"])]],fmt="o",capsize=4,label=chart_label(name))
        ax.legend(fontsize=8,loc="lower right")
    ax.set(xlabel="Номинальные рубли за один правильный ответ",ylabel="Доля верных на 36 задачах (95% ДИ)",
           xlim=(0,None),ylim=(0,1.05),title="Стоимость и точность облачных моделей")
    ax.grid(alpha=.22)
    fig.tight_layout()
    return fig


def local_resources():
    """Сохранённый снимок Ollama; не измерение пиковой памяти процесса."""
    path=lab.DATA/"local_metadata.json"
    if not path.exists():
        return pd.DataFrame(columns=["модель","дата снимка","память модели, байт","VRAM, байт","VRAM, ГиБ","контекст"])
    metadata=json.loads(path.read_text(encoding="utf-8-sig"))
    rows=[]
    for model in metadata.get("running",{}).get("models",[]):
        vram=model.get("size_vram")
        rows.append({"модель":model.get("name",model.get("model")),"дата снимка":metadata.get("timestamp"),
                     "память модели, байт":model.get("size"),"VRAM, байт":vram,
                     "VRAM, ГиБ":vram/2**30 if vram is not None else np.nan,
                     "контекст":model.get("context_length")})
    return pd.DataFrame(rows)
