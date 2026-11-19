"""Эталонные программы D. Никогда не включать в запрос к испытуемой LLM."""
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine, load_iris, make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, average_precision_score


def solve(id):
    cv = StratifiedKFold(3, shuffle=True, random_state=42)
    if id == "D01":
        X,y=load_breast_cancer(return_X_y=True)
        pipeline=make_pipeline(StandardScaler(),SVC(kernel="rbf"))
        grid={"svc__C":[.1,1,10],"svc__gamma":["scale",.01]}
        test_size=.2
    elif id == "D02":
        X,y=load_diabetes(return_X_y=True)
        pipeline=make_pipeline(StandardScaler(),Ridge())
        grid={"ridge__alpha":[.1,1,10]}
        test_size=.2
        cv=KFold(3,shuffle=True,random_state=42)
    elif id == "D03":
        X,y=make_classification(n_samples=1500,n_features=10,n_informative=5,
                               n_redundant=2,weights=[.95,.05],random_state=42)
        a,b,c,d=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
        model=RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=1).fit(a,c)
        return float(average_precision_score(d,model.predict_proba(b)[:,1]))
    elif id == "D04":
        X,y=load_wine(return_X_y=True)
        pipeline=make_pipeline(StandardScaler(),PCA(),LogisticRegression(max_iter=2000,random_state=42))
        grid={"pca__n_components":[2,5,10],"logisticregression__C":[.1,1]}
        test_size=.25
    elif id == "D05":
        X,y=load_iris(return_X_y=True)
        pipeline=make_pipeline(StandardScaler(),KNeighborsClassifier())
        grid={"kneighborsclassifier__n_neighbors":[3,5,9]}
        test_size=.25
    else:
        raise KeyError(id)
    a,b,c,d=train_test_split(X,y,test_size=test_size,random_state=42,
                            stratify=None if id=="D02" else y)
    search=GridSearchCV(pipeline,grid,cv=cv,n_jobs=1,
                        scoring="neg_root_mean_squared_error" if id=="D02" else "accuracy").fit(a,c)
    pred=search.predict(b)
    return float(mean_squared_error(d,pred)**.5 if id=="D02" else accuracy_score(d,pred))


if __name__ == "__main__":
    import json
    print(json.dumps({f"D{i:02}":solve(f"D{i:02}") for i in range(1,6)},indent=2))
