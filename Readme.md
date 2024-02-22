# LegalBench Data


## Summary of the problem

LegalBench is a benchmark consisting of different legal reasoning tasks. Each task has
an associated dataset, consisting of input-output pairs, aiming to develop and evalu-
ate models for various tasks, including text classification, task classification, and text
generation. The dataset comprises 161 tasks with 307 unique answers, designed for
evaluating Legal Language Models (LLMs). The goal is to assess the effectiveness of
LLMs in answering legal questions based on contracts and other legal documents. Ini-
tial steps include exploratory data analysis (EDA) to understand the dataset’s structure
and characteristics, followed by experimenting with different models such as TFIDF
embedding with Gradient Boosting, BERT for classification, and transformer models for
text generation. Future steps involve exploring rule-based approaches and knowledge
graphs to further enhance the understanding and analysis of legal contracts.

```
Figure 1: LegalBench
```
LegalBench tasks are organized into six categories based on the type of legal reasoning
the task requires. These are:

- Issue-spotting: tasks in which an LLM must determine if a set of facts raise a
    particular set of legal questions, implicate an area of the law, or are relevant to a
    specific party.
- Rule-recall: tasks which require the LLM to generate the correct legal rule on an
    issue in a jurisdiction (e.g., the rule for hearsay in US federal court), or answer a
    question about what the law in a jurisdiction does/does not permit.
- Rule-recall: tasks which require the LLM to generate the correct legal rule on an
    issue in a jurisdiction (e.g., the rule for hearsay in US federal court), or answer a
    question about what the law in a jurisdiction does/does not permit.
- Rule-application: tasks which evaluate whether an LLM can explain reasoning in
    a manner which exhibits the correct legal inferences.
- Rule-conclusion: tasks which require an LLM to determine the legal outcome of a
    set of facts under a specified rule.
- Interpretation: tasks which require the LLM to parse and understand a legal text
    (e.g., classifying contractual clauses).
- Rhetorical-understanding: tasks which require an LLM to reason about legal ar-
    gumentation and analysis (e.g., identifying textualist arguments).

```
Figure 2: A sample row from LegalBench task type consumercontractsqa
```
**Some key observations after performing Exploratory Data Analysis** :

1. The LegalBench dataset comprises 161 tasks, each with varying amounts of asso-
    ciated data. The distribution of data across tasks shows a skew, with some tasks
    having significantly more data compared to others.
2. Overall, the dataset contains 307 unique responses, primarily categorical, with
    most tasks having fewer than 10 categories. The majority of questions in the
    dataset elicit Yes/No responses.
3. The word count in the dataset is substantial, ranging from approximately 50 to
    2,000-5,000 words in the prompts.
4. Similarly, question lengths vary from 500 to 5,000 words, generally shorter than
    the prompts.


#### TASKS

**The next step was to define the nature of the problem. We needed to determine
whether it falls under text classification or text generation. For instance, we could
treat the 155 tasks with categorical answers (such as Yes, No, Relevant, or Irrele-
vant) as a text classification problem. Alternatively, we could approach all tasks as
text generation, where the model generates an answer based on a given context.
Additionally, these tasks could be seen as a form of question-answering system,
encompassing text classification, task classification, and text generation.**

### Task classification

**Summary**

LegalBench encompasses 161 tasks, ranging from abercrombie to consumer contracts
q&a and citation predictions [1]. The primary objective is classification: given a prompt,
determine its corresponding task. To accomplish this, we first generate embeddings of
the prompt text column. Initially, I employed the TFIDF vectorizer, converting text
into numerical TFIDF scores. Subsequently, I explored Bert embeddings and the Bert
transformer for text classification tasks. Leveraging the distill Bert model classifier from
HuggingFace, I achieved a promising score of 0.94 on the test dataset (20% split). This
model is conveniently accessible on the HuggingFace hub, enabling users to input a
random prompt and receive the predicted task it aligns with.

**Results & Observations**

- The experimental results demonstrate the superior performance of the BERT model
    over the Gradient Boosting Classifier in task classification. The BERT model achieves
    an impressive test score of 92.3%, while the Gradient Boosting Classifier lags be-
    hind at 63.6%. This disparity suggests that the BERT model excels in capturing
    the intricacies of text and accurately determining the task at hand.
- One plausible reason for this performance gap could be attributed to the BERT
    model’s access to a significantly larger dataset compared to the Gradient Boosting
    Classifier. While the BERT model benefited from training on over 3 million text
    samples, the Gradient Boosting Classifier was limited to a dataset of just 1000
    text samples. This discrepancy in dataset size allows the BERT model to grasp
    more generalizable patterns and comprehend the relationships between words
    and phrases more effectively.
- Another possible explanation for the difference in performance is that the BERT
    model is able to use contextual information to determine the meaning of words.
    The Gradient Boosting Classifier, on the other hand, only uses the individual words

in the text to make its predictions. This can lead to errors when the meaning of a
word depends on the context in which it is used.
```
- Overall, the results of the experiments suggest that the BERT model is a promising
    tool for task classification. The model is able to achieve high accuracy on a variety
    of tasks and is able to learn from a large dataset.


```
Figure 3: Task classifier BERT model on HF
```

### Answer classification

**Summary**

The next task involves classifying answers based on the corresponding question. While
this could be further segmented to classify responses for each task individually, for
this case, we’ll simply label each answer using Label Encoding. The model will then
classify responses based on the prompt. Initially, I used TFIDF vectorization with Gradi-
ent Boosting trees, which surprisingly performed well, achieving a test score of 0.71.
However, the model exhibited severe overfitting, indicated by a higher train score.
This can be addressed by employing a cross-validation approach or by enriching the
dataset with more features, such as identifying if a contract is involved or if the prompt
contains legal terms related to the task. I also examined the confusion matrix for
some tasks to understand their performance in classifying responses. The model’s
performance varies, with scores ranging from as low as 0.56 to a perfect 1. For in-
stance, the task ”cuadrofrroforofn” achieved the lowest score of 0.56, while the task
”maudabilitytoconsummateconceptissubjecttomaecarveouts” scored perfectly. Tran-
sitioning to transformers, I experimented with distill BERT embeddings and a BERT
classifier, achieving an impressive score of 0.94 on the test data.

```
Figure 4: Confusion matrix of few tasks on answer classification
```

### Text Generation (Q&A)

**Results & Observations**

1. Utilizing TFIDF and Gradient Boosting, we achieved a test score of 0.71, indicating
    a slight overfitting from the train score.
2. To potentially enhance the results, different vectorization techniques or hyperpa-
    rameter tuning could be explored. Specifically, for the ”consumercontractsqa”
    category, deeper analysis revealed subpar performance on certain question types,
    suggesting a need for additional data or model refinement.
3. Notably, even when the model predicts incorrectly, it typically remains within the
    task’s answer category. For instance, if the answer should be a simple ”Yes” or
    ”No,” the model generally avoids predicting answers that are entirely unrelated to
    these categories.
4. Upon analyzing the ”consumercontractqa” results, no discernible correlation was
    found between accuracy and the volume of data available for that task in our
    dataset. This implies that the model may perform well even with fewer data
    points.
5. Contrastingly, the BERT model significantly outperformed the Gradient Boosting
    Classifier in task classification, achieving a remarkable test score of 92.3% com-
    pared to the latter’s 63.6%. This indicates that the BERT model is adept at captur-
    ing text nuances and determining the correct task.
6. One potential explanation for this discrepancy lies in the dataset sizes used for
    training. The BERT model trained on over 3 million text samples, while the Gra-
    dient Boosting Classifier trained on just 1000 samples. The larger dataset enables
    the BERT model to learn more generalized patterns and understand word and
    phrase relationships better.
7. Another factor contributing to the performance gap is the BERT model’s ability
    to leverage contextual information. In contrast, the Gradient Boosting Classifier
    relies solely on individual words in the text for predictions, which can lead to
    errors when word meanings depend on context.
8. Overall, these experiments suggest that the BERT model holds promise for task
    classification, boasting high accuracy across various tasks and demonstrating ef-
    fective learning from a large dataset.

In the latter part of the project, the focus shifted to treating the problem as a text
generation task, specifically a question-answering task. This led to the discovery of a
significant paper [3] that detailed training a tiny-Llama model for chat-like applications.
Given that the tiny-Llama model shared the same architecture as the Llama model, it
was a natural fit. The process began with loading the dataset, ensuring that the prompt
followed specific formatting requirements for the Llama model. While the formatting
didn’t affect the eventual embedding, it was advisable to use the ”###” approach to
designate key sections. In this case, three critical sections needed to be passed to the
model: ”Instruction,” ”Prompt/Context,” and ”Response.” The ”Instruction” outlined the
model’s task, such as reading a contract and answering a question based on it. To man-
age the model’s quantized representation, the bits&bytes library was employed, with
Lora being used for efficient model training. Additionally, to monitor the model’s per-
formance and weights, the W&B (Weights & Biases) dashboard was enabled, providing
easy access to experiment logs.
A sample prompt:

Below is an instruction that describes a task, paired with an input that
provides further context.
Write a response that appropriately completes the request.

### Instruction:
Will Google always allow me to transfer my content out of my Google account?

### Input:
Were constantly developing new technologies and features to improve our
services. For example, we invest in artificial intelligence that uses machine
learning to detect and block spam and malware, and to provide you with
innovative features like simultaneous translations.
As part of this continual improvement, we sometimes add or remove features
and functionalities, increase or decrease limits to our services,
and start offering new services or stop offering old ones.

If we make material changes that negatively impact your use of our services
or if we stop offering a service, well provide you with reasonable advance
notice and an opportunity to export your content from your Google Account
using Google Takeout, except in urgent situations such as preventing abuse,
responding to legal requirements or addressing security and operability issues.

``
Experiment Number of classes Classification type Model Train score Test score
1 61 Answer GBD + TFIDF 0.886 0.
2 61 Answer BERT 0.52 0.
3 161 Task GBD + TFIDF 0.85 0.
4 161 Task BERT 0.940 0.
```
```
Table 2: Experiments and results
```

#### CHALLENGES & LEARNING’S

```
Figure 5: A sample output from Llama
```
## Challenges & Learning’s

- A significant challenge in this task was comprehensively identifying all aspects of
    the problem. The dataset comprised 161 tasks, each unique and potentially seen
    as a subtask. It was crucial to categorize and understand the achievable goals
    within this vast dataset.
- Setting up the environments for these experiments posed another challenge. Google
    Colab’s strict policy allowed only one active session with a GPU, limiting the abil-
    ity to run multiple experiments simultaneously, especially those requiring hours
    of training and inference.
- The most valuable takeaway from the project was gaining a deep understanding of
    the LegalBench dataset and how it can effectively evaluate Legal Language Models
    (LLMs).
- An essential learning experience was evaluating the model’s performance and in-
    terpreting the results. This involved exploring how the model performed on each
    task and investigating why some tasks yielded better results than others.
- Additionally, there was a learning curve in prompt engineering and data format-
    ting. This involved preparing the data in a specific way to use as prompts for the
    LLMs, aligning with the format of the data on which these models were trained,
    as detailed in research papers, to achieve optimal results.

## Next Steps

1. Utilize BERT model-generated embeddings for visualization and pattern recog-
    nition, employing dimensionality reduction techniques to identify clusters and
    trends in the data.
2. Experiment with a broader range of machine learning models, including those
    specifically trained on legal text, available from the HuggingFace model hub.
3. Explore zero-shot classification with GPT models using LegalBench as prompts,
    leveraging the model’s ability to generalize to unseen tasks.
4. Investigate the integration of knowledge graphs to efficiently answer questions

## References

```
[1] Guha, Neel and Ho, Daniel E and Nyarko, Julian and R ́e, Christopher, Legalbench:
Prototyping a collaborative benchmark for legal reasoning. pages
```
```
[2] Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M. A., Lacroix, T., ... &
Lample, G. (2023). Llama: Open and efficient foundation language models. arXiv
preprint arXiv:2302.13971.. pages
```
```
[3] Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training
of deep bidirectional transformers for language understanding. arXiv preprint
arXiv:1810.04805.. pages
```
```
[4] Vimal, Bhartendoo. (2020). Application of Logistic Regression in Natural Lan-
guage Processing. International Journal of Engineering Research and. V9.
10.17577/IJERTV9IS060095.. pages
```
```
[5] Wang, S., & Jiang, J. (2015). Learning natural language inference with LSTM.
arXiv preprint arXiv:1512.08849.. pages
```
```
[6] Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training
of deep bidirectional transformers for language understanding. arXiv preprint
arXiv:1810.04805.. pages
```
```
[7] Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., & Amodei,
D. (2020). Language models are few-shot learners. Advances in neural informa-
tion processing systems, 33, 1877-1901.. pages
```
```
[8] Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly
learning to align and translate. arXiv preprint arXiv:1409.0473.. pages
```

