"""Авторский банк: эталоны рассчитываются, а не извлекаются из ответов LLM."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def bayes(p, sensitivity, false_positive):
    return p * sensitivity / (p * sensitivity + (1 - p) * false_positive)


def gini(a, b):
    return 1 - (a / (a + b)) ** 2 - (b / (a + b)) ** 2


def gain(parent, left, right):
    return gini(*parent) - sum(left) / sum(parent) * gini(*left) - sum(right) / sum(parent) * gini(*right)


def confusion(tp, fp, fn, tn):
    return [(tp + tn) / (tp + fp + fn + tn), tp / (tp + fp), tp / (tp + fn), 2 * tp / (2 * tp + fp + fn)]


def task(id, type, topic, ru, en, reference, *, source="Собственная задача, подготовлена с помощью LLM; эталон проверяется кодом/студентом",
         difficulty=2, answer_kind="numeric", solution="", rubric="", cluster=None, rtol=.01, atol=1e-8):
    return dict(id=id, type=type, topic=topic, text_ru=ru, text_en=en,
                reference=reference, answer_kind=answer_kind,
                tolerance={"rtol": rtol, "atol": atol}, source=source,
                difficulty=difficulty, solution=solution, rubric=rubric,
                cluster=cluster or id)


def make_bank():
    tasks = [
        task("A01", "A", "Байесовские методы",
             "Дефект есть у 1% деталей. Чувствительность датчика 95%, вероятность срабатывания без дефекта 5%. Датчик сработал. Найдите вероятность дефекта; ответ дайте долей.",
             "1% of parts are defective. Sensitivity is 95% and the false-positive rate is 5%. The sensor is positive. Find the posterior probability of a defect as a fraction.",
             bayes(.01, .95, .05), source="Адаптировано из примера методички ЛР3, тип A", solution="0.01*0.95/(0.01*0.95+0.99*0.05)", cluster="pair01"),
        task("A02", "A", "Деревья и ансамбли",
             "В родительском узле 40 объектов класса 0 и 10 класса 1. Левый узел (35;2), правый (5;8). Найдите уменьшение взвешенного критерия Джини.",
             "A parent node contains 40 class-0 and 10 class-1 objects; children are (35,2) and (5,8). Find the weighted decrease in Gini impurity.",
             gain((40,10),(35,2),(5,8)), source="Адаптировано из примера методички ЛР3, тип A", solution="G(40,10)-37/50*G(35,2)-13/50*G(5,8)", cluster="pair02"),
        task("A03", "A", "Метрики качества",
             "TP=30, FP=10, FN=20, TN=940. Найдите [accuracy, precision, recall, F1] для положительного класса в таком порядке, долями.",
             "TP=30, FP=10, FN=20, TN=940. Return [accuracy, precision, recall, F1] for the positive class, as fractions.",
             confusion(30,10,20,940), source="Адаптировано из примера методички ЛР3, тип A", solution="[970/1000,30/40,30/50,60/90]", cluster="pair03"),
        task("A04", "A", "SVM",
             "У канонической разделяющей гиперплоскости SVM w=(3,4), b=-2. Классы -1 и +1. Для x=(1,0) найдите [полную ширину зазора между w*x+b=±1, предсказанный класс].",
             "A canonical SVM has w=(3,4), b=-2 and labels -1,+1. For x=(1,0), return [full margin width between w*x+b=±1, predicted label].",
             [.4,1], source="Адаптировано из примера методички ЛР3, тип A", solution="2/||w||=0.4; f(x)=1, класс +1. Точка на границе зазора, не обязательно опорная.", cluster="pair04"),
        task("A05", "A", "Логистическая регрессия",
             "Логистическая регрессия даёт логит z=ln(3). Найдите P(y=1), ответ долей.",
             "A logistic regression produces logit z=ln(3). Find P(y=1) as a fraction.",
             .75, source="Стандартный учебный шаблон функции sigmoid; формулировка собственная", solution="1/(1+exp(-ln(3)))=3/4", cluster="pair05"),
        task("A06", "A", "kNN",
             "Обучающие пары (x,y): (0,0),(2,1),(5,1). Для x=0.5 найдите предсказание kNN при k=1, евклидовой метрике и равных весах.",
             "Training (x,y) pairs are (0,0),(2,1),(5,1). Predict at x=0.5 using kNN with k=1, Euclidean distance and uniform weights.",
             0, source="Стандартный учебный шаблон kNN; формулировка собственная", solution="Ближайшая точка 0 имеет класс 0.", cluster="pair06"),
        task("A07", "A", "PCA",
             "Ковариационная матрица центрированных данных diag(9,4,1). PCA без стандартизации. Какую долю дисперсии объясняет первая компонента?",
             "Centered data have covariance diag(9,4,1). PCA is applied without standardization. What fraction of variance does PC1 explain?",
             9/14, source="Стандартный учебный шаблон PCA; формулировка собственная",
             solution="lambda_max / trace = 9/14", cluster="pair07"),
        task("A08", "A", "Кластеризация",
             "Один кластер состоит из (0,0),(2,0),(0,2). Найдите [координата x центроида, координата y центроида, сумма квадратов евклидовых расстояний до центроида].",
             "A cluster contains (0,0),(2,0),(0,2). Return [centroid x, centroid y, sum of squared Euclidean distances to centroid].",
             [2/3,2/3,16/3], source="Стандартный учебный шаблон k-means; формулировка собственная",
             solution="Среднее=(2/3,2/3); SSE=16/3.", cluster="pair08"),
        task("B01", "B", "PCA",
             "Верно ли: PCA без использования меток выбирает направления, которые лучше всего разделяют классы? Обоснуйте.",
             "True or false: PCA without labels selects directions that best separate classes. Justify.",
             False, answer_kind="boolean", source="Адаптировано из примера методички ЛР3, тип B",
             rubric="False. PCA максимизирует дисперсию проекций, а не разделимость классов."),
        task("B02", "B", "SVM и деревья",
             "Верно ли: положительное линейное масштабирование отдельных признаков обычно не меняет множество допустимых пороговых разбиений точного дерева, но может изменить RBF-SVM с фиксированными C и gamma? Обоснуйте.",
             "True or false: positive linear rescaling of individual features normally preserves the available splits of an exact threshold tree, but can change RBF-SVM with fixed C and gamma. Justify.",
             True, answer_kind="boolean", source="Адаптировано из примера методички ЛР3, тип B",
             rubric="True. Порядок сохраняется у дерева; евклидовы расстояния ядра меняются. Допустимы оговорки о численной точности."),
        task("B03", "B", "Байесовские методы",
             "Верно ли: наивный Байес гарантированно выдаёт калиброванные вероятности на любых данных, потому что он вероятностный? Обоснуйте.",
             "True or false: Naive Bayes guarantees calibrated probabilities on any data because it is probabilistic. Justify.",
             False, answer_kind="boolean", source="Адаптировано из примера методички ЛР3, тип B",
             rubric="False. Нарушение условной независимости/ошибочная спецификация могут давать чрезмерную уверенность."),
        task("B04", "B", "Линейная регрессия",
             "Верно ли: при полном столбцовом ранге X функция ||Xw-y||² имеет единственный минимум по w? Обоснуйте.",
             "True or false: if X has full column rank, ||Xw-y||² has a unique minimizer in w. Justify.",
             True, answer_kind="boolean", rubric="True. X^T X положительно определена; строго выпуклая квадратичная функция."),
        task("B05", "B", "Кластеризация",
             "Верно ли: при увеличении k глобально минимальная обучающая inertia k-means не может увеличиться, если k не больше числа объектов? Обоснуйте.",
             "True or false: increasing k cannot increase the globally minimal training inertia of k-means, for k not exceeding the sample size. Justify.",
             True, answer_kind="boolean", rubric="True. Дополнительный центр может разделить кластер; речь о глобальном оптимуме, не случайном запуске."),
        task("B06", "B", "Метрики качества",
             "Верно ли: ROC-AUC=0.95 само по себе гарантирует precision не ниже 0.8 при пороге 0.5? Обоснуйте.",
             "True or false: ROC-AUC=0.95 alone guarantees precision at least 0.8 at threshold 0.5. Justify.",
             False, answer_kind="boolean", rubric="False. ROC-AUC оценивает ранжирование, precision зависит от порога и базовой частоты."),
        task("C01", "C", "Линейные методы",
             "150 размеченных виброзаписей, 400 спектральных признаков, 3 класса. Требуется интерпретируемая модель. Предложите метод, регуляризацию и честную оценку; несколько записей могут относиться к одному подшипнику.",
             "150 labeled vibration records, 400 spectral features, 3 classes. Interpretability is required. Propose a method, regularization and valid evaluation; several records may belong to the same bearing.",
             "Регуляризованная линейная модель с групповой CV",
             answer_kind="rubric", source="Адаптировано и расширено из примера методички ЛР3, тип C",
             rubric="2: регуляризованная логистическая/линейная SVM или обоснованная разреженная альтернатива; scaling внутри Pipeline/CV; группировка по подшипнику; объяснение p>n и коэффициентов."),
        task("C02", "C", "Метрики и ансамбли",
             "В 100000 измерениях станка только 0.5% отказов. Пропуск отказа дорог. Нужно сравнить простую модель и бустинг. Какие метрики, порог и разбиение выбрать, если измерения идут по времени?",
             "Only 0.5% of 100000 machine records are failures. Missing a failure is costly. Compare a simple model and boosting. Choose metrics, a threshold and a split for temporal records.",
             "PR-AUC, стоимость/recall, порог по validation, временное разбиение",
             answer_kind="rubric", rubric="2: PR-AUC и recall/стоимость; временной holdout; порог только по validation; константный baseline; исключение недоступных на прогнозе признаков."),
        task("C03", "C", "Кластеризация",
             "Нужно группировать точки геолокации: кластеры невыпуклые, есть шум, число кластеров неизвестно. Предложите метод и объясните метрику расстояния и выбор параметров без истинных меток.",
             "Cluster geolocation points: clusters are non-convex, noise is present, and the number of clusters is unknown. Propose a method, distance metric and unlabeled parameter selection.",
             "DBSCAN/HDBSCAN с географической метрикой",
             answer_kind="rubric", rubric="2: DBSCAN/HDBSCAN или обоснованный плотностный метод; haversine с радианами/подходящая проекция; eps/min_samples или аналоги, анализ устойчивости и шума. Не выбирать k-means без оговорок."),
        task("C04", "C", "Байесовские методы",
             "5000 размеченных коротких сообщений, десятки тысяч разреженных счётчиков слов. Нужен быстрый baseline на CPU. Выберите классификатор, представление и схему оценки. Дубликаты сообщений встречаются.",
             "5000 labeled short messages, tens of thousands of sparse word-count features. A fast CPU baseline is needed. Choose a classifier, representation and evaluation; duplicate messages exist.",
             "MultinomialNB/ComplementNB или линейная модель",
             answer_kind="rubric", rubric="2: подходящий sparse NB/линейный классификатор; CountVectorizer/TF-IDF обучается только на train; дубликаты не между split; стратификация/группы; метрика согласована с балансом."),
        task("C05", "C", "PCA и регрессия",
             "300 образцов сплава, 120 сильно коррелированных числовых признаков, целевая — прочность. Инженер предлагает выбрать PCA по всему датасету, затем оценить регрессию CV. Что исправить и какие две модели сравнить?",
             "300 alloy samples, 120 strongly correlated numeric features, target is strength. An engineer proposes fitting PCA on all data before regression CV. Correct the protocol and propose two models.",
             "Pipeline scaling/PCA/regression внутри CV; Ridge против PCR",
             answer_kind="rubric", rubric="2: утечка при PCA/scaling вне CV; вложенный подбор только train; Ridge и PCA+регрессия (или обоснованный PLS); RMSE/MAE; PCA не оптимизирует связь с y."),
    ]
    d_specs = [
        ("D01","SVM","На load_breast_cancer предскажите исходный target. train_test_split(test_size=0.2, stratify=y, random_state=42). Pipeline StandardScaler+SVC(kernel='rbf'). Подберите C из [0.1,1,10] и gamma из ['scale',0.01] по accuracy на StratifiedKFold(3,shuffle=True,random_state=42), n_jobs=1. Выведите test accuracy.",
         "On load_breast_cancer, predict the original target. Split test_size=0.2, stratify=y, random_state=42. Use StandardScaler+RBF SVC in a Pipeline. Tune C=[0.1,1,10], gamma=['scale',0.01] by accuracy with StratifiedKFold(3,shuffle=True,random_state=42), n_jobs=1. Print test accuracy.", .02),
        ("D02","Линейная регрессия","На load_diabetes разделите данные: test_size=0.2, random_state=42. Pipeline StandardScaler+Ridge. Подберите alpha=[0.1,1,10] с KFold(3,shuffle=True,random_state=42), scoring='neg_root_mean_squared_error', n_jobs=1. Выведите test RMSE.",
         "Split load_diabetes with test_size=0.2, random_state=42. Use StandardScaler+Ridge in a Pipeline. Tune alpha=[0.1,1,10] with KFold(3,shuffle=True,random_state=42), scoring='neg_root_mean_squared_error', n_jobs=1. Print test RMSE.", 2.0),
        ("D03","Деревья и метрики","Сгенерируйте make_classification(n_samples=1500,n_features=10,n_informative=5,n_redundant=2,weights=[0.95,0.05],random_state=42). Разбейте test_size=0.2,stratify=y,random_state=42. Обучите RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=1). Выведите average_precision по predict_proba для класса 1, без подбора по тесту.",
         "Generate make_classification(n_samples=1500,n_features=10,n_informative=5,n_redundant=2,weights=[0.95,0.05],random_state=42). Split test_size=0.2,stratify=y,random_state=42. Fit RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=1). Print class-1 average_precision using predict_proba, without test tuning.", .03),
        ("D04","PCA и логистическая регрессия","На load_wine разделите test_size=0.25,stratify=y,random_state=42. Pipeline StandardScaler,PCA,LogisticRegression(max_iter=2000,random_state=42). Подберите pca__n_components=[2,5,10], logisticregression__C=[0.1,1] по accuracy, StratifiedKFold(3,shuffle=True,random_state=42), n_jobs=1. Выведите test accuracy.",
         "Split load_wine with test_size=0.25,stratify=y,random_state=42. Use a StandardScaler,PCA,LogisticRegression(max_iter=2000,random_state=42) pipeline. Tune pca__n_components=[2,5,10], logisticregression__C=[0.1,1] by accuracy, StratifiedKFold(3,shuffle=True,random_state=42), n_jobs=1. Print test accuracy.", .03),
        ("D05","kNN","На load_iris разделите test_size=0.25,stratify=y,random_state=42. Pipeline StandardScaler+KNeighborsClassifier. Подберите n_neighbors=[3,5,9] по accuracy, StratifiedKFold(3,shuffle=True,random_state=42), n_jobs=1. Выведите test accuracy.",
         "Split load_iris with test_size=0.25,stratify=y,random_state=42. Use StandardScaler+KNeighborsClassifier in a Pipeline. Tune n_neighbors=[3,5,9] by accuracy, StratifiedKFold(3,shuffle=True,random_state=42), n_jobs=1. Print test accuracy.", .03),
    ]
    for id, topic, ru, en, tol in d_specs:
        tasks.append(task(id,"D",topic,ru,en,{"implementation":f"reference_d.py::{id}","metric_atol":tol},
                          answer_kind="code", rubric="См. rubrics.md; параметры и split должны соответствовать условию."))
    tasks.extend([
        task("F01","F","Байесовские методы",
             "Дефект есть у 20% деталей. Чувствительность датчика 95%, вероятность срабатывания без дефекта 5%. Датчик сработал. Найдите вероятность дефекта долей.",
             "20% of parts are defective. Sensitivity is 95%, false-positive rate is 5%. The sensor is positive. Find the posterior defect probability as a fraction.",
             bayes(.2,.95,.05), solution=".2*.95/(.2*.95+.8*.05)",cluster="pair01"),
        task("F02","F","Деревья и ансамбли",
             "В родительском узле 30 объектов класса 0 и 20 класса 1. Левый узел (25;5), правый (5;15). Найдите уменьшение взвешенного критерия Джини.",
             "A parent node contains 30 class-0 and 20 class-1 objects; children are (25,5) and (5,15). Find the weighted decrease in Gini impurity.",
             gain((30,20),(25,5),(5,15)),solution="G(30,20)-30/50*G(25,5)-20/50*G(5,15)",cluster="pair02"),
        task("F03","F","Метрики качества",
             "TP=30, FP=20, FN=10, TN=940. Найдите [accuracy, precision, recall, F1] для положительного класса в таком порядке, долями.",
             "TP=30, FP=20, FN=10, TN=940. Return [accuracy, precision, recall, F1] for the positive class as fractions.",
             confusion(30,20,10,940),solution="[970/1000,30/50,30/40,60/90]",cluster="pair03"),
        task("F04","F","SVM",
             "У канонической разделяющей гиперплоскости SVM w=(6,8), b=-4. Классы -1 и +1. Для x=(0,0) найдите [полную ширину зазора между w*x+b=±1, предсказанный класс].",
             "A canonical SVM has w=(6,8), b=-4 and labels -1,+1. At x=(0,0), return [full margin width between w*x+b=±1, predicted label].",
             [.2,-1],solution="2/10=.2; f=-4, класс -1.",cluster="pair04"),
        task("F05","F","Логистическая регрессия",
             "Логистическая регрессия даёт логит z=-ln(3). Найдите P(y=1), ответ долей.",
             "A logistic regression produces logit z=-ln(3). Find P(y=1) as a fraction.",
             .25,solution="1/(1+exp(ln(3)))=1/4",cluster="pair05"),
        task("F06","F","kNN",
             "Обучающие пары (x,y): (0,0),(2,1),(5,1). Для x=0.5 найдите предсказание kNN при k=3, евклидовой метрике и равных весах.",
             "Training (x,y) pairs are (0,0),(2,1),(5,1). Predict at x=0.5 using kNN with k=3, Euclidean distance and uniform weights.",
             1,solution="Голоса [0,1,1], большинство 1.",cluster="pair06"),
        task("F07","F","Метрики качества",
             "У бинарного классификатора accuracy=0.9 на 1000 объектах. Найдите precision положительного класса. Если нельзя — ответьте INSUFFICIENT_DATA.",
             "A binary classifier has accuracy=0.9 on 1000 objects. Find positive-class precision. If it cannot be determined, answer INSUFFICIENT_DATA.",
             "INSUFFICIENT_DATA",answer_kind="label",solution="Числа TP и FP не заданы.",difficulty=1),
        task("F08","F","SVM",
             "SVM имеет b=2, точка x=(1,1), классы -1/+1. Вектор w не указан. Определите предсказанный класс; если нельзя — INSUFFICIENT_DATA.",
             "An SVM has b=2, point x=(1,1), labels -1/+1, but w is not supplied. Predict its label or answer INSUFFICIENT_DATA.",
             "INSUFFICIENT_DATA",answer_kind="label",solution="Знак w*x+b зависит от неизвестного w.",difficulty=1),
        task("F09","F","Метрики качества",
             "В обучении 8000 строк, 120 признаков и 300 деревьев. На тесте TP=18,FP=6,FN=12,TN=164. Найдите F1 положительного класса, долей.",
             "Training uses 8000 rows, 120 features and 300 trees. Test counts are TP=18,FP=6,FN=12,TN=164. Find positive-class F1 as a fraction.",
             2/3,solution="2TP/(2TP+FP+FN)=36/54; размеры train не нужны."),
        task("F10","F","Линейная регрессия",
             "Для фиксированной линейной регрессии на тесте y=[2,4,6], прогноз=[3,3,6]. Модель обучали 12 секунд на 700 строках. Найдите MSE.",
             "A fixed linear regression has test y=[2,4,6] and predictions=[3,3,6]. Training took 12 seconds on 700 rows. Find MSE.",
             2/3,solution="(1+1+0)/3; время и число train строк не нужны."),
        task("F11","F","PCA",
             "Ковариационная матрица центрированных данных diag(16,3,1). PCA без стандартизации. Какую долю дисперсии объясняет первая компонента?",
             "Centered data have covariance diag(16,3,1). PCA is applied without standardization. What fraction of variance does PC1 explain?",
             .8,solution="16/(16+3+1)=.8",cluster="pair07"),
        task("F12","F","Кластеризация",
             "Один кластер состоит из (0,0),(3,0),(0,3). Найдите [координата x центроида, координата y центроида, сумма квадратов евклидовых расстояний до центроида].",
             "A cluster contains (0,0),(3,0),(0,3). Return [centroid x, centroid y, sum of squared Euclidean distances to centroid].",
             [1,1,12],solution="Среднее=(1,1); SSE=2+5+5=12.",cluster="pair08"),
        task("E01","E","LLM как классификатор",
             "Отдельный протокол: load_breast_cancer, 100 тестовых строк, 0/5/20 примеров. Положительный класс — malignancy=1. См. ноутбук.",
             "Separate protocol: load_breast_cancer, 100 test rows, 0/5/20 demonstrations. Positive class is malignancy=1. See notebook.",
             {"dataset":"sklearn.load_breast_cancer","test_size":100,"shots":[0,5,20],"seed":42},answer_kind="experiment",
             rubric="F1 редкого положительного класса, balanced accuracy, majority accuracy; единые split и примеры.")
    ])
    return tasks


def pilots():
    return [
        task("P01","A","Метрики качества","TP=8,FP=2. Найдите precision долей.","TP=8,FP=2. Find precision as a fraction.",.8),
        task("P02","B","PCA","PCA использует метки классов? Ответ true или false с объяснением.","Does PCA use class labels? Answer true or false and explain.",False,answer_kind="boolean"),
        task("P03","C","Регрессия","Нужен простой прогноз вещественной цены по числовым признакам. Предложите baseline.","Suggest a simple baseline for predicting real-valued prices from numeric features.","Линейная регрессия",answer_kind="rubric"),
        task("P04","D","sklearn","Напишите код: загрузите load_iris и выведите число строк. Один Python-блок.","Write Python code to load load_iris and print the row count. One Python block.",150,answer_kind="code"),
        task("P05","F","Метрики качества","Известно только TP=7. Найдите recall, либо ответьте INSUFFICIENT_DATA.","Only TP=7 is known. Find recall, or answer INSUFFICIENT_DATA.","INSUFFICIENT_DATA",answer_kind="label"),
    ]


def write_bank():
    for filename, records in [("tasks.jsonl",make_bank()),("pilots.jsonl",pilots())]:
        path = ROOT / filename
        text = "".join(json.dumps(x,ensure_ascii=False)+"\n" for x in records)
        if path.exists() and path.read_text(encoding="utf-8") != text and (ROOT/"data/preregistration.lock.json").exists():
            raise RuntimeError("Протокол уже заморожен: банк не перезаписывается.")
        path.write_text(text,encoding="utf-8")


if __name__ == "__main__":
    write_bank()
