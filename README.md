# Perceptron (Keras) — Zara Fashion Sales (Kaggle)

> **Objetivo da atividade**  
> Desenvolver um **Perceptron** (modelo sequencial Keras com **uma única camada `Dense(1, activation="sigmoid")**), treinar por **50 épocas** com **batch size = 10**, **optimizer = adam**, **loss = binary_crossentropy**, avaliar com **accuracy** **e** **F1**, e **interpretar os resultados**.  
> Entregar o **caderno `.ipynb`** em um repositório GitHub.

**Notebook:** [`perceptron_zara_fashion_sales.ipynb`](./perceptron_zara_fashion_sales.ipynb)

---

## Dataset

- Fonte: Kaggle — *Zara Fashion Sales Dataset and Report* (`mohanz123/zara-fashion-sales-dataset-and-report`).
- Após limpeza (CSV separado por `;`), o conjunto tem **252 amostras** e **16 colunas**:
  - **Numéricas:** `price`, `sales_volume`.
  - **Categóricas:** `product_position` (Aisle/End-cap/Front of Store), `product_category` (Clothing), `promotion` (Yes/No), `seasonal` (Yes/No), `brand` (Zara), `section` (MAN/WOMAN), `currency` (USD).
  - **Texto:** `name`, `description`.
  - **Tempo/ID:** `scraped_at`, `product_id`, `sku`, `url`, `terms`.

**Por que não é “toy”?**  
Embora o N seja enxuto, é um problema **real** de varejo, **multimodal** (numérico, categórico e texto), com **colinearidades** e **sazonalidade**. Isso impõe desafios que vão além de *toy datasets* clássicos (Iris, Pima).

---

## Tarefa de Classificação

Definimos um alvo binário **`high_revenue`** por critério **data-driven**:

- **Receita**: `revenue = price × sales_volume`  
- **Classe positiva**: `high_revenue = 1` se `revenue` > **mediana**; caso contrário `0`.

> Essa formulação evita vazamentos óbvios (não prevemos receita contínua) e produz classes razoavelmente equilibradas.

---

## Modelo

- **Arquitetura**: `Sequential([Dense(1, activation="sigmoid")])`  
  → fronteira **linear** (equivalente à regressão logística).
- **Otimizador**: **Adam** — gradiente adaptativo por parâmetro, converge bem sem grande tuning.
- **Função de perda**: **Binary Crossentropy** — adequada para probabilidade binária com `sigmoid`.
- **Métricas**:
  - **Accuracy** — proporção de acertos globais.
  - **F1** — harmônica de precision/recall (robusta a leve desbalanceamento).

**Treino**: **50 épocas**, **batch_size = 10**, validação interna para monitorar *overfitting*.

---

## Como rodar

> Se usar Python 3.12, o TF pode falhar. Alternativa: `pip install keras>=3.3.0` e usar backend NumPy (Keras Core).

### Execução

* Abra o notebook [`perceptron_zara_fashion_sales.ipynb`](./perceptron_zara_fashion_sales.ipynb).
* A célula de carregamento usa **KaggleHub**; caso necessário, informe o `file_path` correto do CSV (ou leia com `sep=";"`).

---

## Exploração do Dataset (resumo)

* **Amostras**: 252 | **Features**: 16
* **Distribuições** (exemplos):

  * `promotion`: 47,6% Yes / 52,4% No
  * `section`: MAN 218 / WOMAN 34
  * `product_position`: Aisle 97 / End-cap 86 / Front of Store 69
  * `price`: média ≈ 86,25 (USD); **sales\_volume**: média ≈ 1.824

---

## Resultados (Perceptron — obrigatório)

**Teste**:

* **Accuracy**: **0.8824**
* **F1**: **0.8750**

**Matriz de confusão**:

```
[[24  2]
 [ 4 21]]
```

**Interpretação**
O Perceptron, mesmo com fronteira linear, capturou bem o sinal de alta receita (duas variáveis-chave são diretamente relacionadas ao alvo). Falsos positivos foram baixos; falsos negativos moderados. O desempenho sugere que, para este dataset, uma fronteira linear + pré-processamento simples já fornece um **baseline sólido**.

---

## Ir além da ponderada

Para ir além do esperado, fiz três movimentos com a hipótese de que (i) o limiar padrão **0,5** raramente é o ponto ótimo para **F1**, (ii) nomes e descrições carregam sinais semânticos úteis que não aparecem nas colunas estruturadas e (iii) existem **interações não lineares** (ex.: `preço × sazonalidade`, `posição × promoção`) que um Perceptron linear não captura.
Assim, ajustei o **threshold** por validação (otimizando F1), agreguei **TF-IDF** de `name/description` e testei um **MLP raso** (L2 + Dropout) controlando overfitting.

**Comparativo resumido (teste):**

* **Perceptron (baseline estruturadas)** → Acc **0.8824**, F1 **0.8750**
* **Perceptron + TF-IDF + tuning** → Acc **0.7647**, F1 **0.7391**
* **MLP raso + TF-IDF + tuning** → Acc **0.8627**, F1 **0.8571**

**Leitura**: Texto “cru” (alta dimensionalidade) ajudou o **MLP** a ficar competitivo, mas **não superou** o baseline linear bem ajustado neste N. Próximo passo com melhor custo-benefício: **reduzir dimensionalidade do TF-IDF (TruncatedSVD/LSA)** antes do MLP.

---

## Limitações & melhorias

* **Tamanho do dataset** (252 linhas) limita a expressividade de modelos mais complexos.
* **Próximos passos**:

  1. **Threshold tuning** orientado a objetivo (Fβ se recall da classe 1 for mais importante).
  2. **Engenharia de features** (interações simples, *binning* de preço/volume).
  3. **Texto condensado** (TF-IDF → SVD 50–100 comps) antes de MLP.
  4. **Validação K-fold** para estimativa mais estável.

---

## Estrutura do repositório

```
.
├── perceptron_zara_fashion_sales.ipynb   # Notebook principal (EDA, alvo, modelo, treino, avaliação)
└── README.md                             # Este documento
```

---

## Referências técnicas (breve)

* **Adam**: otimizador com momentos adaptativos (aprendizado por parâmetro).
* **Binary Crossentropy**: mede a divergência entre `y_true ∈ {0,1}` e `y_pred ∈ (0,1)`.
* **F1**: balanço entre *precision* e *recall*, útil quando erros têm custos assimétricos.
