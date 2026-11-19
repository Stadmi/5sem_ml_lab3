"""Инструменты эксперимента. Импорт не выполняет сетевых запросов."""
from __future__ import annotations
import hashlib
import itertools
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from credential_utils import valid_gigachat_authorization_key
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                             f1_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
SEED = 42
H1_PAIRS = [(f"A{i:02}",f"F{i:02}") for i in range(1,7)] + [("A07","F11"),("A08","F12")]
PROTOCOL_FILES = ["tasks.jsonl", "pilots.jsonl", "models.json", "preregistration.md",
                  "rubrics.md", "bank.py", "reference_d.py", "lab.py",
                  "local/Qwen3-4B-nothink.Modelfile",
                  "local/Qwen3-4B-nothink.template.txt"]
load_dotenv(ROOT / ".env", override=False)


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else canonical(value).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def records(path):
    path = Path(path)
    if not path.exists():
        return []
    # Не скрываем повреждённые/частично записанные строки.
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_record(row):
    DATA.mkdir(exist_ok=True)
    with (DATA / "responses.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def config():
    return read_json(ROOT / "models.json")


def gigachat_verify():
    """Явно заданный CA либо проектный bundle; остальные провайдеры не затронуты."""
    explicit=os.getenv("GIGACHAT_CA_BUNDLE")
    if explicit:
        path=Path(explicit)
        return str(path if path.is_absolute() else ROOT/path)
    bundled=DATA/"certificates/gigachat-ca-bundle.pem"
    return str(bundled) if bundled.exists() else True


def pricing_for(model):
    """Тарифы — отдельные датированные исходные данные анализа, не параметры генерации."""
    path=ROOT/"tariffs.json"
    if path.exists():
        override=read_json(path).get("models",{}).get(model["name"])
        if override:
            return override
    return model["pricing"]


def duration_seconds(text):
    if not text:
        return 0.
    text=str(text).strip()
    if re.fullmatch(r"\d+(?:\.\d+)?",text):
        return float(text)
    units={"h":3600.,"m":60.,"s":1.,"ms":.001}
    return sum(float(value)*units[unit] for value,unit in
               re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)",text))


class QuotaPause(RuntimeError):
    """Остановиться, не переходя на оплату и не повторяя отклонённый запрос."""


def bank():
    return records(ROOT / "tasks.jsonl")


def inventory():
    rows = []
    for m in config()["models"]:
        t = m["transport"]
        needed = (["GIGACHAT_CREDENTIALS"] if t == "gigachat" else
                  ["YANDEX_API_KEY", "YANDEX_FOLDER_ID"] if t == "yandex" else
                  [m["key_env"]] if t == "openai_compatible" and m.get("key_env") else [])
        rows.append({"name":m["name"], "role":m["role"], "model":m["model"],
                     "enabled":m["enabled"], "credentials_present":all(bool(os.getenv(k)) for k in needed),
                     "missing_variables":", ".join(k for k in needed if not os.getenv(k))})
    return pd.DataFrame(rows)


def validate_bank(tasks=None):
    tasks = bank() if tasks is None else tasks
    ids = [t["id"] for t in tasks]
    assert len(ids) == len(set(ids)) and 30 <= len(ids) <= 40
    counts = pd.Series([t["type"] for t in tasks]).value_counts()
    assert all(counts.get(k,0) >= 5 for k in "ABCDF") and counts.get("E",0) >= 1
    assert sum(t["source"].startswith("Собственная") for t in tasks) >= math.ceil(len(tasks)/2)
    b = [t for t in tasks if t["type"] == "B"]
    assert sum(t["reference"] is True for t in b) == len(b)/2
    assert all(t["text_ru"] and t["text_en"] and t["source"] for t in tasks)
    lookup = {t["id"]:t for t in tasks}
    assert all(lookup[a]["cluster"] == lookup[f]["cluster"] for a,f in H1_PAIRS)
    return counts.sort_index()


def protocol_hashes():
    return {name:digest((ROOT/name).read_bytes()) for name in PROTOCOL_FILES}


def freeze_protocol(*, confirmed=False):
    if not confirmed:
        raise RuntimeError("Сначала проверьте банк, модели и прогноз, затем явно confirmed=True.")
    validate_bank()
    cfg = config()["models"]
    if len(cfg) != 4 or not all(m["enabled"] for m in cfg):
        raise RuntimeError("Сначала настройте и включите ровно четыре модели.")
    expected = {"russian_1","russian_2","foreign_large","local_small"}
    if {m["role"] for m in cfg} != expected:
        raise RuntimeError("Не выполнен состав четырёх моделей.")
    lock = DATA/"preregistration.lock.json"
    if lock.exists():
        ensure_frozen()
        return read_json(lock)
    git = subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
    dirty = subprocess.run(["git","status","--porcelain","--",*PROTOCOL_FILES],cwd=ROOT,
                           capture_output=True,text=True,check=True).stdout
    if dirty.strip():
        raise RuntimeError("Сначала зафиксируйте все файлы протокола коммитом.")
    value = {"created_at":now(),"git_commit":git,"hashes":protocol_hashes(),"confirmed":True}
    write_json(lock,value)
    return value


def ensure_frozen():
    lock = DATA/"preregistration.lock.json"
    if not lock.exists():
        raise RuntimeError("Нет предрегистрации. Сначала модели, пилот, коммит и freeze_protocol.")
    value = read_json(lock)
    if value["hashes"] != protocol_hashes():
        raise RuntimeError("Протокол изменён после заморозки. Не продолжайте старый эксперимент.")
    return value


class Experiment:
    def __init__(self, *, allow_remote=False):
        self.cfg = config()
        self.allow_remote = allow_remote
        self._giga = None
        self._last = {}
        self._verified_local_digest = None
        history=records(DATA/"responses.jsonl")
        self._cache={}
        self._rate={}
        for row in history:
            if row.get("status")=="ok":
                self._cache.setdefault(row["cache_key"],row)
            headers=row.get("rate_limit_headers")
            if headers and row.get("elapsed_s") is not None:
                end=datetime.fromisoformat(row["timestamp"]).timestamp()+row["elapsed_s"]
                age=max(0,time.time()-end)
                self._rate[row["model"]]={"headers":{k.lower():v for k,v in headers.items()},
                                          "at":time.monotonic()-age}

    def _wait_for_quota(self,model,messages,phase):
        state=self._rate.get(model["name"])
        interval=model.get("min_interval_seconds_by_phase",{}).get(
            phase,model.get("min_interval_seconds",2.1))
        wait=max(0,interval-(time.monotonic()-self._last.get(model["name"],0)))
        if state:
            headers=state["headers"]
            age=time.monotonic()-state["at"]
            # Консервативная оценка резерва, не подмена фактического usage.
            reserve=math.ceil(sum(len(msg["content"]) for msg in messages)/2)+self.cfg["max_output_tokens"]
            for dimension in ("tokens","requests"):
                remaining=headers.get("x-ratelimit-remaining-"+dimension)
                if remaining is None:
                    continue
                required=reserve if dimension=="tokens" else 1
                try:
                    insufficient=float(remaining)<required
                except ValueError:
                    continue
                if insufficient:
                    reset=duration_seconds(headers.get("x-ratelimit-reset-"+dimension))
                    if reset>age:
                        wait=max(wait,reset-age+.25)
        maximum=model.get("max_auto_wait_seconds",90)
        if wait>maximum:
            raise QuotaPause(f"{model['name']}: квота требует паузы около {math.ceil(wait)} с; API-запрос не отправлен.")
        if wait>0:
            time.sleep(wait)  # Соблюдение квот до запроса, не повтор неудачной генерации.

    def _giga_token(self):
        if self._giga and self._giga[1] > time.time() + 30:
            return self._giga[0]
        credential = os.environ["GIGACHAT_CREDENTIALS"]
        if not valid_gigachat_authorization_key(credential):
            raise RuntimeError("GigaChat: нужен ключ авторизации Base64(client_id:client_secret), а не Client Secret или access_token.")
        # TLS никогда не выключается; при необходимости задайте доверенный CA bundle.
        response = requests.post("https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
            headers={"Authorization":f"Basic {credential}","RqUID":str(uuid.uuid4())},
            data={"scope":os.getenv("GIGACHAT_SCOPE","GIGACHAT_API_PERS")},
            verify=gigachat_verify(), timeout=60)
        if not response.ok:
            raise RuntimeError(f"GigaChat OAuth HTTP {response.status_code}")
        obj = response.json()
        self._giga = (obj["access_token"], float(obj["expires_at"])/1000)
        return self._giga[0]

    def ask(self, model, prompt, temperature=None, *, task_id, phase="bank"):
        """Возвращает словарь с ответом, usage и временем; каждый вызов логируется."""
        m = next(x for x in self.cfg["models"] if x["name"] == model)
        if not m["enabled"]:
            raise RuntimeError(f"{model}: модель выключена в models.json")
        if phase not in {"smoke","pilot","bank","E"}:
            raise ValueError(phase)
        if phase in {"bank","E"}:
            lock = ensure_frozen()
        else:
            lock = None
        if m["transport"] not in {"ollama","ollama_generate"} and not self.allow_remote:
            raise RuntimeError("Сеть к провайдерам требует явного allow_remote=True.")
        temp = m["temperature"] if temperature is None else temperature
        if phase in {"bank","E"} and temp != m["temperature"]:
            raise RuntimeError("Температура отличается от замороженного протокола.")
        messages = [{"role":"system","content":self.cfg["system_prompt"]},
                    {"role":"user","content":prompt}]
        identity = {"model_config":m,"messages":messages,"phase":phase,"task_id":task_id,
                    "temperature":temp,"max_tokens":self.cfg["max_output_tokens"]}
        key = digest(identity)
        if key in self._cache:
            return self._cache[key]
        self._wait_for_quota(m,messages,phase)
        row = {"request_id":str(uuid.uuid4()),"cache_key":key,"timestamp":now(),"model":model,
               "configured_model":m["model"],"role":m["role"],"phase":phase,"task_id":task_id,
               "temperature":temp,"messages":messages,"configuration_hash":digest(m),
               "max_output_tokens":self.cfg["max_output_tokens"],
               "protocol_commit":lock["git_commit"] if lock else None,
               "status":"error","response":None,"usage":{},"effective_model":None,
               "model_version":None,"finish_reason":None}
        append_record(dict(row,status="started",elapsed_s=None))
        start = time.perf_counter()
        try:
            t = m["transport"]
            if t in {"ollama","ollama_generate"}:
                expected = m.get("expected_digest")
                if expected and self._verified_local_digest != expected:
                    tags=requests.get(m["base_url"].rstrip("/")+"/api/tags",timeout=10)
                    tags.raise_for_status()
                    actual=next((entry.get("digest") for entry in tags.json().get("models",[])
                                 if entry.get("name")==m["model"]),None)
                    if actual != expected:
                        raise RuntimeError("Локальная модель отличается от зафиксированного digest.")
                    self._verified_local_digest=actual
                row["model_version"]=expected
                body = {"model":m["model"],"stream":False,
                        "think":m.get("think",False),
                        "options":{"temperature":temp,"seed":SEED,"num_predict":self.cfg["max_output_tokens"],
                                   "num_ctx":m.get("num_ctx",16384)}}
                if t=="ollama_generate":
                    template=(ROOT/m["template_file"]).read_text(encoding="utf-8")
                    rendered=template.format(system=messages[0]["content"],user=prompt)
                    body.update(prompt=rendered,raw=True)
                    row["rendered_prompt"]=rendered
                    endpoint="/api/generate"
                else:
                    body["messages"]=messages
                    endpoint="/api/chat"
                row["generation_parameters"] = {k:v for k,v in body.items() if k not in {"messages","prompt"}}
                response=requests.post(m["base_url"].rstrip("/")+endpoint,json=body,timeout=600)
            elif t == "yandex":
                uri=m["model"].replace("{folder_id}",os.environ["YANDEX_FOLDER_ID"])
                body={"modelUri":uri,"completionOptions":{"stream":False,"temperature":temp,
                       "maxTokens":self.cfg["max_output_tokens"]},
                      "messages":[{"role":x["role"],"text":x["content"]} for x in messages]}
                response=requests.post(m["base_url"].rstrip("/")+"/completion",json=body,
                    headers={"Authorization":"Api-Key "+os.environ["YANDEX_API_KEY"],
                             "x-folder-id":os.environ["YANDEX_FOLDER_ID"]},timeout=300)
            elif t in {"gigachat","openai_compatible"}:
                token=self._giga_token() if t=="gigachat" else os.environ[m["key_env"]]
                body={"model":m["model"],"messages":messages,"temperature":temp,
                      "max_tokens":self.cfg["max_output_tokens"],"stream":False}
                response=requests.post(m["base_url"].rstrip("/")+"/chat/completions",json=body,
                    headers={"Authorization":"Bearer "+token},
                    verify=gigachat_verify() if t=="gigachat" else True,timeout=300)
            else:
                raise ValueError("Неизвестный transport")
            row["http_status"]=response.status_code
            row["rate_limit_headers"]={
                k:v for k,v in response.headers.items()
                if k.lower().startswith("x-ratelimit-") or k.lower()=="retry-after"}
            if not response.ok:
                # Не логируем заголовки авторизации или произвольное тело ошибки.
                row["retry_after"]=response.headers.get("Retry-After")
                raise RuntimeError(f"{model}: HTTP {response.status_code}; повтор не выполнялся")
            obj=response.json()
            if t in {"ollama","ollama_generate"}:
                row.update(response=obj["response"] if t=="ollama_generate" else obj["message"]["content"],
                    reasoning=obj.get("thinking") if t=="ollama_generate" else obj["message"].get("thinking"),
                    effective_model=obj.get("model"),
                    usage={"prompt_tokens":obj.get("prompt_eval_count"),"completion_tokens":obj.get("eval_count")},
                    finish_reason=obj.get("done_reason"),server_total_duration_ns=obj.get("total_duration"))
            elif t=="yandex":
                result=obj["result"]
                alt=result["alternatives"][0]
                use=result.get("usage",{})
                row.update(response=alt["message"]["text"],effective_model=uri,
                    model_version=result.get("modelVersion"),
                    usage={"prompt_tokens":int(use["inputTextTokens"]) if use.get("inputTextTokens") is not None else None,
                           "completion_tokens":int(use["completionTokens"]) if use.get("completionTokens") is not None else None},
                    finish_reason=alt.get("status"))
            else:
                choice=obj["choices"][0]
                provider_usage=obj.get("usage",{})
                usage=dict(provider_usage)
                if t=="gigachat":
                    # GigaChat возвращает cached input отдельно от prompt_tokens.
                    # Сохраняем исходные поля и считаем физический объём входа.
                    cached=provider_usage.get("precached_prompt_tokens",0) or 0
                    if provider_usage.get("prompt_tokens") is not None:
                        usage["prompt_tokens"]=provider_usage["prompt_tokens"]+cached
                    usage["prompt_tokens_details"]={"cached_tokens":cached}
                    if provider_usage.get("total_tokens") is not None:
                        usage["total_tokens"]=provider_usage["total_tokens"]+cached
                row.update(response=choice["message"].get("content") or "",usage=usage,
                    provider_usage=provider_usage,
                    effective_model=obj.get("model",m["model"]),finish_reason=choice.get("finish_reason"),
                    system_fingerprint=obj.get("system_fingerprint"))
            row["status"]="ok"
        except Exception as exc:
            # Тип исключения и наши безопасные HTTP-сообщения, но не секреты.
            row["error_type"]=type(exc).__name__
            row["error"]=str(exc) if isinstance(exc,RuntimeError) else "Проверьте подключение/настройки; секреты исключены из журнала."
        except (KeyboardInterrupt,SystemExit):
            row["error_type"]="Interrupted"
            row["error"]="Запуск прерван до получения результата."
            raise
        finally:
            row["elapsed_s"]=time.perf_counter()-start
            self._last[model]=time.monotonic()
            if row.get("rate_limit_headers"):
                self._rate[model]={"headers":{k.lower():v for k,v in row["rate_limit_headers"].items()},
                                   "at":time.monotonic()}
            append_record(row)
        if row["status"] != "ok":
            raise RuntimeError(f"Запрос записан как ошибка: {row['error_type']}; см. журнал.")
        self._cache[key]=row
        return row

    def local_metadata(self):
        """Вызывать после smoke отдельно: сохранение ответа не задерживается диагностикой."""
        m=next(m for m in self.cfg["models"] if m["transport"] in {"ollama","ollama_generate"})
        show=requests.post(m["base_url"].rstrip("/")+"/api/show",json={"model":m["model"]},timeout=10)
        show.raise_for_status()
        ps=requests.get(m["base_url"].rstrip("/")+"/api/ps",timeout=10)
        ps.raise_for_status()
        value={"timestamp":now(),"model":m["model"],"show":show.json(),"running":ps.json(),
               "note":"Размер/VRAM загруженной модели — снимок, не измерение пиковой памяти."}
        write_json(DATA/"local_metadata.json",value)
        return value

    def run_bank(self, *, phase="bank", model_names=None, on_result=None):
        if phase not in {"bank","pilot"}:
            raise ValueError(phase)
        tasks=bank() if phase=="bank" else records(ROOT/"pilots.jsonl")
        names=model_names or [m["name"] for m in self.cfg["models"] if m["enabled"]]
        for task in tasks:
            if task["type"]=="E":
                continue
            for name in names:
                row=self.ask(name,task["text_ru"],task_id=task["id"],phase=phase)
                if on_result:
                    on_result(row)
        return response_frame(phase)


def response_frame(phase="bank"):
    rows=[r for r in records(DATA/"responses.jsonl") if r["phase"]==phase and r["status"]=="ok"]
    if not rows:
        return pd.DataFrame()
    frame=pd.DataFrame(rows)
    # Первый ответ; старые протоколы/изменённые промпты не смешиваются.
    if phase in {"bank","E"} and frame.duplicated(["model","task_id"]).any():
        raise RuntimeError("Найдено несколько разных ответов на одну задачу. Разделите версии эксперимента.")
    return frame


def final_text(text, finish_reason=None):
    """Отделить финал от явного/предзаполненного think; исходный лог не меняется."""
    if finish_reason in {"length","max_tokens","ALTERNATIVE_STATUS_TRUNCATED"}:
        return None
    if "</think>" in text:
        text=text.rsplit("</think>",1)[1]
    if "<think>" in text:
        return None
    return text


def parse_answer(text, finish_reason=None):
    text=final_text(text,finish_reason)
    if text is None:
        return {"value":None,"confidence":None,"parse_ok":False}
    # \s содержит перевод строки: нельзя захватывать УВЕРЕННОСТЬ как пустой ОТВЕТ.
    answers=[v.strip() for v in re.findall(r"(?im)^[ \t]*ОТВЕТ:[ \t]*([^\r\n]+)$",text) if v.strip()]
    conf=re.findall(r"(?im)^[ \t]*УВЕРЕННОСТЬ:[ \t]*(\d+(?:[.,]\d+)?)[ \t]*%?[ \t\r]*$",text)
    confidence=float(conf[-1].replace(",","."))/100 if conf else None
    if confidence is not None and not 0<=confidence<=1:
        confidence=None
    if not answers:
        return {"value":None,"confidence":confidence,"parse_ok":False}
    raw=answers[-1].strip().strip("`")
    try:
        value=json.loads(raw)
    except ValueError:
        if re.fullmatch(r"[+-]?\d+(?:[.,]\d+)?(?:[eE][+-]?\d+)?\s*%?",raw):
            value=float(raw.rstrip("% ").replace(",",".")) / (100 if raw.endswith("%") else 1)
        else:
            value=raw.strip('"').strip("'")
    return {"value":value,"confidence":confidence,"parse_ok":True}


def correct(task, value):
    if task["answer_kind"] in {"rubric","code","experiment"}:
        # Даже обрезанный C/D требует предусмотренной протоколом ручной оценки.
        return None
    if value is None:
        return False
    kind=task["answer_kind"]
    ref=task["reference"]
    if kind=="numeric":
        if isinstance(value,bool):
            return False
        try:
            a,b=np.asarray(value,dtype=float),np.asarray(ref,dtype=float)
            return bool(a.shape==b.shape and np.isfinite(a).all() and
                        np.allclose(a,b,rtol=task["tolerance"]["rtol"],atol=task["tolerance"]["atol"]))
        except (ValueError,TypeError):
            return False
    if kind=="boolean":
        if isinstance(value,str):
            value={"true":True,"false":False,"верно":True,"неверно":False}.get(value.lower().strip())
        return type(value) is bool and value==ref
    if kind=="label":
        return isinstance(value,str) and value.strip().upper()==ref
    return None


def automatic_scores():
    frame=response_frame()
    if frame.empty:
        return pd.DataFrame()
    lookup={t["id"]:t for t in bank()}
    rows=[]
    for r in frame.to_dict("records"):
        task=lookup[r["task_id"]]
        parsed=parse_answer(r["response"],r.get("finish_reason"))
        result=correct(task,parsed["value"])
        rows.append({"model":r["model"],"task_id":task["id"],"type":task["type"],
            "cluster":task["cluster"],"correct":result,**parsed,
            "elapsed_s":r["elapsed_s"],"prompt_tokens":r["usage"].get("prompt_tokens"),
            "completion_tokens":r["usage"].get("completion_tokens"),
            "abstained":bool(isinstance(parsed["value"],str) and
                (parsed["value"].upper()=="INSUFFICIENT_DATA" or
                 any(x in parsed["value"].lower() for x in ["недостаточно данных","данных недостаточно","не могу","неизвестно"])))})
    return pd.DataFrame(rows)


def export_blind_reviews():
    frame=response_frame()
    if frame.empty:
        return "Нет ответов для проверки."
    lookup={t["id"]:t for t in bank()}
    samples=[]
    keys=[]
    for r in frame.to_dict("records"):
        t=lookup[r["task_id"]]
        if t["type"] not in "BCD":
            continue
        sample_id=digest(r["request_id"])[:16]
        samples.append(dict(sample_id=sample_id,type=t["type"],task=t["text_ru"],
            rubric=t["rubric"],answer=r["response"],score="",pass1="",result_ok="",
            methodology_error="",notes=""))
        keys.append(dict(sample_id=sample_id,model=r["model"],task_id=t["id"]))
    if not samples:
        return "Нет B/C/D."
    anonymous=pd.DataFrame(samples).sample(frac=1,random_state=SEED)
    for n in (1,2):
        path=DATA/f"reviews_{n}.csv"
        if path.exists():
            raise FileExistsError(f"{path.name} уже существует: оценки не перезаписываются.")
    pd.DataFrame(keys).to_csv(DATA/"reviews_key.csv",index=False,encoding="utf-8-sig")
    for n in (1,2):
        anonymous.to_csv(DATA/f"reviews_{n}.csv",index=False,encoding="utf-8-sig")
    return len(samples)


def manual_scores():
    paths=[DATA/"reviews_key.csv",DATA/"reviews_1.csv",DATA/"reviews_2.csv"]
    if not all(p.exists() for p in paths):
        return pd.DataFrame(),{}
    key,a,b=[pd.read_csv(p) for p in paths]
    for frame in (key,a,b):
        if frame["sample_id"].duplicated().any():
            raise ValueError("Дубликаты sample_id")
    for frame in (a,b):
        if set(frame["sample_id"])!=set(key["sample_id"]):
            raise ValueError("Набор проверяемых ответов изменён.")
    merged=key.merge(a,on="sample_id",validate="one_to_one").merge(
        b,on="sample_id",suffixes=("_1","_2"),validate="one_to_one")
    kappas={}
    out=[]
    for kind in "BCD":
        group=merged[merged["type_1"]==kind].copy()
        if group.empty or group[["score_1","score_2"]].isna().any().any():
            continue
        for col in ("score_1","score_2"):
            if not group[col].isin([0,1,2]).all():
                raise ValueError("Баллы только 0,1,2.")
        k=float(cohen_kappa_score(group["score_1"],group["score_2"]))
        # При полном совпадении единственной категории κ математически не определена.
        kappas[kind]={"kappa":k if np.isfinite(k) else None,
                      "agreement":float((group.score_1==group.score_2).mean()),"n":len(group)}
        if np.isfinite(k) and k<.6:
            raise RuntimeError(f"κ для {kind} < .6: уточните рубрику и переоцените.")
        for r in group.to_dict("records"):
            if kind=="D":
                values=[r[f"{col}_{n}"] for n in (1,2) for col in ("pass1","result_ok","methodology_error")]
                if not all(pd.notna(v) and v in (0,1) for v in values):
                    continue
                x=bool(r["pass1_1"] and r["result_ok_1"] and not r["methodology_error_1"])
                y=bool(r["pass1_2"] and r["result_ok_2"] and not r["methodology_error_2"])
            else:
                x,y=r["score_1"]==2,r["score_2"]==2
            if x!=y:
                raise RuntimeError("Есть несогласованная correctness: сохраните исходные оценки и согласуйте.")
            out.append({"model":r["model"],"task_id":r["task_id"],"manual_correct":x,
                        "score_1":r["score_1"],"score_2":r["score_2"],
                        **({c:r[f"{c}_1"] for c in ("pass1","result_ok","methodology_error")} if kind=="D" else {})})
    return pd.DataFrame(out),kappas


def bootstrap_ci(values, *, n=1000, seed=SEED, clusters=None):
    arr=np.asarray(values,dtype=float)
    if not len(arr) or not np.isfinite(arr).all():
        raise ValueError("Нужны конечные наблюдения.")
    rng=np.random.default_rng(seed)
    if clusters is None:
        means=arr[rng.integers(0,len(arr),(n,len(arr)))].mean(axis=1)
    else:
        labels=np.asarray(clusters)
        unique=np.unique(labels)
        chunks=[arr[labels==u] for u in unique]
        means=np.array([np.concatenate([chunks[i] for i in rng.integers(0,len(chunks),len(chunks))]).mean() for _ in range(n)])
    return tuple(float(x) for x in np.percentile(means,[2.5,97.5]))


def calibration(correctness, confidence):
    y=np.asarray(correctness,dtype=float)
    c=np.asarray(confidence,dtype=float)
    if len(y)!=len(c) or not len(y) or not np.isfinite(c).all() or ((c<0)|(c>1)).any():
        raise ValueError("Некорректные данные калибровки.")
    idx=np.minimum((c*5).astype(int),4)
    bins=[]
    for i in range(5):
        take=idx==i
        bins.append({"bin":i,"left":i/5,"right":(i+1)/5,"n":int(take.sum()),
                     "confidence":float(c[take].mean()) if take.any() else None,
                     "accuracy":float(y[take].mean()) if take.any() else None})
    ece=sum(b["n"]/len(y)*abs(b["confidence"]-b["accuracy"]) for b in bins if b["n"])
    return float(ece),pd.DataFrame(bins)


def compare_models(frame):
    pivot=frame.pivot(index="task_id",columns="model",values="correct")
    rows=[]
    for a,b in itertools.combinations(pivot.columns,2):
        pair=pivot[[a,b]].dropna().astype(bool)
        both=int((pair[a]&pair[b]).sum())
        only_a=int((pair[a]&~pair[b]).sum())
        only_b=int((~pair[a]&pair[b]).sum())
        neither=int((~pair[a]&~pair[b]).sum())
        rows.append({"model_a":a,"model_b":b,"n":len(pair),"both":both,"only_a":only_a,
                     "only_b":only_b,"neither":neither,
                     "p":float(mcnemar([[both,only_a],[only_b,neither]],exact=True).pvalue)})
    out=pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"]=multipletests(out["p"],method="holm")[1]
    return out


def dataset_e():
    ds=load_breast_cancer()
    X=ds.data
    y=(ds.target==0).astype(int)
    checksum=digest(X.tobytes()+y.tobytes())
    pool,test=train_test_split(np.arange(len(y)),test_size=100,stratify=y,random_state=SEED)
    twenty,_=train_test_split(pool,train_size=20,stratify=y[pool],random_state=SEED)
    five,_=train_test_split(twenty,train_size=5,stratify=y[twenty],random_state=SEED)
    split={"seed":SEED,"dataset_sha256":checksum,"positive_class":"malignant",
           "test":test.tolist(),"train20":twenty.tolist(),"train5":five.tolist()}
    path=DATA/"e_split.json"
    if path.exists() and read_json(path)!=split:
        raise RuntimeError("Сохранённое разбиение отличается: не перезаписываем.")
    write_json(path,split)
    frame=pd.DataFrame(X,columns=ds.feature_names)
    frame["malignant"]=y
    frame.to_csv(DATA/"e_dataset.csv",index_label="row_id",float_format="%.17g")
    return X,y,ds.feature_names,split


def class_metrics(y,pred):
    y=np.asarray(y)
    pred=np.asarray(pred,dtype=int)
    valid=np.isin(pred,[0,1])
    # Невалидный ответ явно становится ошибкой относительно истинной метки.
    # Это не доступный модели прогноз: консервативное правило расчёта метрик.
    assessed=np.where(valid,pred,1-y)
    return {"f1":float(f1_score(y,assessed,zero_division=0)),
            "balanced_accuracy":float(balanced_accuracy_score(y,assessed)),
            "accuracy":float(accuracy_score(y,assessed)),"valid_rate":float(valid.mean())}


def run_baselines():
    X,y,names,s=dataset_e()
    test=np.array(s["test"])
    rows=[]
    predictions=[]
    for shots in (5,20):
        train=np.array(s[f"train{shots}"])
        for name,model in [
            ("LogisticRegression",make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,random_state=SEED))),
            ("GaussianNB",GaussianNB())]:
            start=time.perf_counter()
            model.fit(X[train],y[train])
            pred=model.predict(X[test])
            rows.append({"model":name,"shots":shots,**class_metrics(y[test],pred),
                         "fit_predict_s":time.perf_counter()-start})
            rows[-1].update(classification_ci(y[test],pred))
            predictions.extend(dict(model=name,shots=shots,row_id=int(i),true=int(t),pred=int(p))
                               for i,t,p in zip(test,y[test],pred))
    dummy=DummyClassifier(strategy="most_frequent").fit(X[s["train20"]],y[s["train20"]])
    pred=dummy.predict(X[test])
    rows.append({"model":"Always frequent class","shots":20,**class_metrics(y[test],pred),"fit_predict_s":None})
    rows[-1].update(classification_ci(y[test],pred))
    predictions.extend(dict(model="Always frequent class",shots=20,row_id=int(i),true=int(t),pred=int(p))
                       for i,t,p in zip(test,y[test],pred))
    result=pd.DataFrame(rows)
    result.to_csv(DATA/"e_baselines.csv",index=False)
    pd.DataFrame(predictions).to_csv(DATA/"e_baseline_predictions.csv",index=False)
    return result


def classification_ci(truth,pred):
    truth,pred=np.asarray(truth),np.asarray(pred)
    rng=np.random.default_rng(SEED)
    ids=[np.flatnonzero(truth==k) for k in (0,1)]
    samples=[]
    for _ in range(1000):
        idx=np.concatenate([rng.choice(ix,len(ix),replace=True) for ix in ids])
        samples.append(class_metrics(truth[idx],pred[idx]))
    result={}
    for key in ("f1","balanced_accuracy","accuracy"):
        result[key+"_ci_low"],result[key+"_ci_high"]=[
            float(v) for v in np.percentile([sample[key] for sample in samples],[2.5,97.5])]
    return result


def e_prompt(row_id,shots):
    X,y,names,s=dataset_e_cached()
    def describe(i):
        # Все 30 исходных float, без округления или отбора признаков.
        return json.dumps(X[i].tolist(),separators=(",",":"))
    prompt=("Определи класс опухоли по числовым признакам исследования. "
            "1 — злокачественная, 0 — доброкачественная. "
            "Имена признаков оставлены в исходном английском виде, единицы и масштабы не менялись. "
            "Каждый объект записан текстом как массив из 30 значений. "
            "Порядок признаков одинаков у всех объектов и задан один раз:\n"+
            json.dumps(names.tolist(),ensure_ascii=False,separators=(",",":"))+"\n")
    for i in s[f"train{shots}"] if shots else []:
        prompt+=f"Пример: {describe(i)}\nКласс: {y[i]}\n"
    return prompt+f"Объект для классификации: {describe(row_id)}\nВерни ОТВЕТ: 0 либо ОТВЕТ: 1 и строку уверенности."


_e_cache=None


def dataset_e_cached():
    global _e_cache
    if _e_cache is None:
        _e_cache=dataset_e()
    return _e_cache


def run_e(experiment, model_names=None, shots_values=(0,5,20), on_result=None):
    ensure_frozen()
    X,y,names,s=dataset_e_cached()
    for name in model_names or [m["name"] for m in config()["models"] if m["enabled"]]:
        for shots in shots_values:
            if shots not in (0,5,20):
                raise ValueError("Допустимы только 0/5/20 примеров.")
            for i in s["test"]:
                row=experiment.ask(name,e_prompt(i,shots),task_id=f"E01:{shots}:{i}",phase="E")
                if on_result:
                    on_result(row)
    return score_e()


def score_e():
    frame=response_frame("E")
    if frame.empty:
        return pd.DataFrame()
    X,y,names,s=dataset_e_cached()
    rows=[]
    for (model,shots),group in frame.assign(shots=frame.task_id.str.split(":").str[1].astype(int)).groupby(["model","shots"]):
        lookup={int(r["task_id"].split(":")[2]):r for r in group.to_dict("records")}
        if set(lookup)!=set(s["test"]):
            continue
        pred=[]
        for i in s["test"]:
            value=parse_answer(lookup[i]["response"],lookup[i].get("finish_reason"))["value"]
            pred.append(int(value) if type(value) in (int,float) and value in (0,1) else -1)
        truth=y[s["test"]]
        row={"model":model,"shots":int(shots),**class_metrics(truth,pred)}
        row.update(classification_ci(truth,pred))
        rows.append(row)
    result=pd.DataFrame(rows)
    if not result.empty:
        result.to_csv(DATA/"e_llm.csv",index=False)
    return result


def score_all():
    auto=automatic_scores()
    if auto.empty:
        return auto,{}
    manual,kappas=manual_scores()
    if not manual.empty:
        auto=auto.merge(manual,on=["model","task_id"],how="left",validate="one_to_one")
        take=auto.type.isin(["C","D"])
        auto.loc[take,"correct"]=auto.loc[take,"manual_correct"]
    auto.to_csv(DATA/"scores.csv",index=False)
    write_json(DATA/"review_agreement.json",kappas)
    return auto,kappas


def hypotheses(scores):
    automatic=scores[scores.type.isin(["A","B","F"]) & scores.correct.notna()].copy()
    result={"H1":[],"H5":None,"H10":[]}
    for model,group in automatic.groupby("model"):
        values=group.set_index("task_id")["correct"]
        a=[a for a,f in H1_PAIRS]
        f=[f for a,f in H1_PAIRS]
        if all(k in values for k in a+f):
            av=values.loc[a].to_numpy(dtype=float)
            fv=values.loc[f].to_numpy(dtype=float)
            only_a=int(((av==1)&(fv==0)).sum())
            only_f=int(((av==0)&(fv==1)).sum())
            p=float(mcnemar([[int(((av==1)&(fv==1)).sum()),only_a],
                              [only_f,int(((av==0)&(fv==0)).sum())]],exact=True).pvalue)
            result["H1"].append({"model":model,"delta":float((av-fv).mean()),
                                 "ci":bootstrap_ci(av-fv),"p":p,"pairs":len(H1_PAIRS)})
        cal=group[group.confidence.notna()]
        expected_cal={t["id"] for t in bank() if t["type"] in "ABF"}
        if len(cal) and set(group.task_id)==expected_cal:
            ece,bins=calibration(cal.correct.astype(float),cal.confidence.astype(float))
            delta=cal.confidence.astype(float)-cal.correct.astype(float)
            ci=bootstrap_ci(delta,clusters=cal.cluster)
            result["H10"].append({"model":model,"delta":float(delta.mean()),"ci":ci,"ece":ece,
                "confidence_coverage":len(cal)/len(group),
                "status":"поддержана" if ci[0]>0 else "противоречие" if ci[1]<0 else "данных недостаточно",
                "exploratory":True})
            bins.to_csv(DATA/f"calibration_{model}.csv",index=False)
    if result["H1"]:
        # Семейство фиксировано на четырёх моделях даже при частичном прогоне.
        adjusted=multipletests([r["p"] for r in result["H1"]]+[1.]*(4-len(result["H1"])),method="holm")[1]
        for row,p in zip(result["H1"],adjusted):
            row["p_holm"]=float(p)
            row["status"]=("поддержана" if row["delta"]>0 else "противоречие") if p<.05 else "данных недостаточно"
    cfg=config()["models"]
    local=next(m["name"] for m in cfg if m["role"]=="local_small")
    large=[m["name"] for m in cfg if m["role"]!="local_small"]
    piv=automatic[automatic.type.isin(["A","B"])].pivot(index="task_id",columns="model",values="correct")
    ids_a=[t["id"] for t in bank() if t["type"]=="A"]
    ids_b=[t["id"] for t in bank() if t["type"]=="B"]
    if set([local,*large]).issubset(piv.columns) and set(ids_a+ids_b).issubset(piv.index):
        piv=piv.loc[ids_a+ids_b,[local,*large]]
        if not piv.isna().any().any():
            gap=piv[large].astype(float).mean(axis=1)-piv[local].astype(float)
            a,b=gap.loc[ids_a].to_numpy(),gap.loc[ids_b].to_numpy()
            rng=np.random.default_rng(SEED)
            boot=a[rng.integers(0,len(a),(1000,len(a)))].mean(1)-b[rng.integers(0,len(b),(1000,len(b)))].mean(1)
            ci=np.percentile(boot,[2.5,97.5]).tolist()
            result["H5"]={"gap_A":float(a.mean()),"gap_B":float(b.mean()),"difference":float(a.mean()-b.mean()),
                          "ci":ci,"status":"поддержана" if ci[0]>0 else "противоречие" if ci[1]<0 else "данных недостаточно"}
    write_json(DATA/"hypotheses.json",result)
    return result


def analyze():
    scores,kappas=score_all()
    if scores.empty:
        return {"status":"Нет ответов LLM. Общие метрики и гипотезы не рассчитаны."}
    report={"status":"Неполные данные","kappas":kappas,"hypotheses":hypotheses(scores)}
    expected={t["id"] for t in bank() if t["type"]!="E"}
    model_names={m["name"] for m in config()["models"]}
    complete=set(scores.model)==model_names and scores.correct.notna().all()
    complete=complete and all(set(g.task_id)==expected for _,g in scores.groupby("model"))
    report["bank_complete"]=bool(complete)
    if not complete:
        report["missing_manual_or_answers"]=int(scores.correct.isna().sum())
        write_json(DATA/"analysis_status.json",report)
        return report
    rows=[]
    for model,group in scores.groupby("model"):
        y=group.correct.astype(float)
        lo,hi=bootstrap_ci(y)
        clo,chi=bootstrap_ci(y,clusters=group.cluster)
        m=next(m for m in config()["models"] if m["name"]==model)
        price=pricing_for(m)
        usage_ok=group[["prompt_tokens","completion_tokens"]].notna().all().all()
        priced=all(price.get(k) is not None for k in ("input_rub_per_million","output_rub_per_million"))
        priced=priced and bool(price.get("source") and price.get("accessed_at"))
        cost=(group.prompt_tokens.sum()*price["input_rub_per_million"]+
              group.completion_tokens.sum()*price["output_rub_per_million"])/1e6 if priced and usage_ok else None
        rows.append({"model":model,"n":len(y),"accuracy":float(y.mean()),"ci_low":lo,"ci_high":hi,
                     "cluster_ci_low":clo,"cluster_ci_high":chi,"refusal_rate":float(group.abstained.mean()),
                     "latency_s":float(group.elapsed_s.mean()),
                     "tokens_mean":float((group.prompt_tokens+group.completion_tokens).mean()) if usage_ok else None,
                     "cost_rub":cost,"cost_per_correct_rub":cost/y.sum() if cost is not None and y.sum()>0 else None})
    summary=pd.DataFrame(rows)
    summary.to_csv(DATA/"summary.csv",index=False)
    compare_models(scores).to_csv(DATA/"mcnemar.csv",index=False)
    scores.assign(correct=scores.correct.astype(float)).pivot_table(
        index="model",columns="type",values="correct",aggfunc="mean").to_csv(DATA/"heatmap.csv")
    d=scores[scores.type=="D"]
    if "pass1" in d:
        d.groupby("model")[["pass1","result_ok","methodology_error"]].mean().to_csv(DATA/"code_metrics.csv")
    report["status"]="Банк оценён. Проверьте полноту E, тарифов и разбора пяти ошибок."
    report["classification_complete"]=len(score_e())==12
    write_json(DATA/"analysis_status.json",report)
    make_plots()
    return report


def make_plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGURES.mkdir(exist_ok=True)
    if (DATA/"summary.csv").exists():
        summary=pd.read_csv(DATA/"summary.csv")
        heat=pd.read_csv(DATA/"heatmap.csv",index_col=0)
        fig,ax=plt.subplots(figsize=(8,4))
        image=ax.imshow(heat.to_numpy(),vmin=0,vmax=1,cmap="YlGnBu")
        ax.set(xticks=range(len(heat.columns)),xticklabels=heat.columns,
               yticks=range(len(heat)),yticklabels=heat.index,title="Точность: модель × тип задачи")
        for i in range(len(heat)):
            for j in range(len(heat.columns)):
                ax.text(j,i,f"{heat.iloc[i,j]:.2f}",ha="center",va="center")
        fig.colorbar(image,ax=ax); fig.tight_layout(); fig.savefig(FIGURES/"heatmap.png",dpi=180); plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,4))
        ax.errorbar(summary.model,summary.accuracy,
                    yerr=np.vstack([summary.accuracy-summary.ci_low,summary.ci_high-summary.accuracy]),
                    fmt="o",capsize=5)
        ax.set(ylabel="Accuracy",ylim=(0,1.05),title="Accuracy и 95% бутстреп-ДИ")
        fig.tight_layout(); fig.savefig(FIGURES/"accuracy_ci.png",dpi=180); plt.close(fig)
        priced=summary[summary.cost_per_correct_rub.notna()]
        if not priced.empty:
            fig,ax=plt.subplots(figsize=(7,4))
            ax.scatter(priced.cost_per_correct_rub,priced.accuracy)
            for r in priced.to_dict("records"):
                ax.annotate(r["model"],(r["cost_per_correct_rub"],r["accuracy"]))
            ax.set(xlabel="Рублей за правильный ответ",ylabel="Accuracy",title="Стоимость — точность")
            fig.tight_layout(); fig.savefig(FIGURES/"cost_accuracy.png",dpi=180); plt.close(fig)
    if (DATA/"e_baselines.csv").exists():
        baseline=pd.read_csv(DATA/"e_baselines.csv")
        fig,ax=plt.subplots(figsize=(8,4))
        for name,group in baseline.groupby("model"):
            ax.plot(group.shots,group.f1,"o-",label=name)
        if (DATA/"e_llm.csv").exists():
            for name,group in pd.read_csv(DATA/"e_llm.csv").groupby("model"):
                ax.plot(group.shots,group.f1,"s--",label=name)
        ax.set(xlabel="Число примеров",ylabel="F1 злокачественного класса",ylim=(0,1.05),title="Тип E: реальные результаты")
        ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIGURES/"classification.png",dpi=180); plt.close(fig)
    hpath=DATA/"hypotheses.json"
    if hpath.exists():
        h=read_json(hpath)
        if h["H1"]:
            fig,ax=plt.subplots(figsize=(7,4))
            ax.bar([r["model"] for r in h["H1"]],[r["delta"] for r in h["H1"]])
            ax.axhline(0,color="black"); ax.set(ylabel="Accuracy исходных − изменённых",title="H1: восемь пар")
            fig.tight_layout(); fig.savefig(FIGURES/"h1.png",dpi=180); plt.close(fig)
        if h["H5"]:
            fig,ax=plt.subplots(figsize=(6,4))
            ax.bar(["Тип A","Тип B"],[h["H5"]["gap_A"],h["H5"]["gap_B"]])
            ax.set(ylabel="Accuracy больших − локальной",title="H5: разрыв по типам")
            fig.tight_layout(); fig.savefig(FIGURES/"h5.png",dpi=180); plt.close(fig)
        if h["H10"]:
            fig,ax=plt.subplots(figsize=(6,5))
            ax.plot([0,1],[0,1],"k--")
            for r in h["H10"]:
                bins=pd.read_csv(DATA/f"calibration_{r['model']}.csv").dropna()
                ax.plot(bins.confidence,bins.accuracy,"o-",label=r["model"])
            ax.set(xlabel="Средняя уверенность",ylabel="Доля верных",xlim=(0,1),ylim=(0,1),title="H10: диаграмма надёжности")
            ax.legend(); fig.tight_layout(); fig.savefig(FIGURES/"h10.png",dpi=180); plt.close(fig)


def environment_snapshot():
    DATA.mkdir(exist_ok=True)
    info={"timestamp":now(),"python":sys.version,"executable":sys.executable,
          "platform":platform.platform(),"logical_cpus":os.cpu_count(),"seed":SEED}
    try:
        import psutil
        info["ram_gb"]=psutil.virtual_memory().total/1024**3
        gpu=subprocess.run(["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader"],
                           capture_output=True,text=True,timeout=10)
        info["gpu"]=gpu.stdout.strip()
    except (ImportError,FileNotFoundError,subprocess.TimeoutExpired):
        pass
    write_json(DATA/"environment.json",info)
    freeze=subprocess.run([sys.executable,"-m","pip","freeze"],capture_output=True,text=True,check=True)
    (ROOT/"requirements.freeze.txt").write_text(freeze.stdout,encoding="utf-8")
    return info
