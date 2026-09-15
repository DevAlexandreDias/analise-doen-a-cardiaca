import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# =========================================================================
# 1. PREPARAÇÃO DOS DADOS E DIVISÃO (TRAIN / TEST)
# =========================================================================
# Define a semente aleatória para garantir a reprodutibilidade
np.random.seed(42)

# Divide os dados (80% para treino e 20% para teste)
# Certifique-se de que as variáveis 'X' e 'y' já estejam carregadas no seu ambiente
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# =========================================================================
# 2. MODELO INDIVIDUAL COM PIPELINE (LinearSVC + StandardScaler)
# =========================================================================
# Padroniza os dados e aplica o LinearSVC em um único fluxo seguro
svc_model = make_pipeline(StandardScaler(), LinearSVC(random_state=42))
svc_model.fit(X_train, y_train)

y_pred_svc = svc_model.predict(X_test)
print(f"Acurácia do LinearSVC: {accuracy_score(y_test, y_pred_svc) * 100:.2f}%\n")

# =========================================================================
# 3. COMPARAÇÃO DE MÚLTIPLOS MODELOS DE CLASSIFICAÇÃO
# =========================================================================
models = {
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
}


def fit_and_score(models, X_train, X_test, y_train, y_test):
    np.random.seed(42)
    model_scores = {}

    for name, model in models.items():
        # Treina o modelo
        model.fit(X_train, y_train)
        # Calcula a acurácia nos dados de teste
        model_scores[name] = model.score(X_test, y_test)

    return model_scores


# Executa o treino comparativo e exibe as notas
 scores_comparacao = fit_and_score(models, X_train, X_test, y_train, y_test)
print("Resultado da Acurácia por Modelo:")
for modelo, pontuacao in scores_comparacao.items():
    print(f"- {modelo}: {pontuacao * 100:.2f}%")

# =========================================================================
# 4. GRÁFICO: DISTRIBUIÇÃO DAS PROBABILIDADES (Regressão Logística)
# =========================================================================
# Instancia e treina a Regressão Logística para extrair probabilidades
log_reg = models["Logistic Regression"]
log_reg.fit(X_train, y_train)

# Extrai a chance (0 a 1) da classe positiva (1 = Com Doença)
y_probs = log_reg.predict_proba(X_test)[:, 1]

plt.figure(figsize=(10, 5))
sns.histplot(
    x=y_probs,
    hue=y_test,
    kde=True,
    bins=20,
    palette={0: "green", 1: "red"},
    element="step",
)
plt.axvline(
    x=0.5,
    color="black",
    linestyle="--",
    linewidth=2,
    label="Fronteira de Decisão (50%)",
)

plt.title(
    "Distribuição da Chance de Doença Cardíaca (Regressão Logística)",
    fontsize=13,
    fontweight="bold",
)
plt.xlabel("Chance Prevista pelo Modelo (0.0 = 0% | 1.0 = 100%)", fontsize=11)
plt.ylabel("Quantidade de Pacientes", fontsize=11)
plt.legend(["Fronteira (50%)", "Com Doença (1)", "Saudável (0)"])
plt.tight_layout()
plt.show()

# =========================================================================
# 5. GRÁFICO: COMPARAÇÃO DE IDADE POR DIAGNÓSTICO (Boxplot)
# =========================================================================
plt.figure(figsize=(8, 5))

# Certifique-se de que a coluna 'age' exista no seu DataFrame X_test
sns.boxplot(
    x=y_test,
    y=X_test["age"],
    palette={0: "green", 1: "red"},
    hue=y_test,
    legend=False,
)

plt.title(
    "Distribuição da Idade por Diagnóstico de Doença Cardíaca",
    fontsize=13,
    fontweight="bold",
)
plt.xlabel("Condição do Paciente", fontsize=11)
plt.ylabel("Idade (Anos)", fontsize=11)
plt.xticks(ticks=[0, 1], labels=["Saudável (0)", "Com Doença (1)"])
plt.tight_layout()
plt.show()
