# Knowledge Tracing Transformer (SQKT)

This repository provides the code for a knowledge tracing model that leverages student code submissions, problem texts, student questions, and skill extraction to predict the likelihood of success on future programming exercises. It integrates a Transformer encoder and CodeT5 for both representation learning and auxiliary generation tasks.

---

## Setup

### Requirements

We recommend using Python 3.8+ and creating a virtual environment.

Install the dependencies via pip:

```bash
pip install -r requirements.txt
```

---

## Data Format

The model requires four types of CSV files per dataset. These are usually located under a folder like `data/`.

### 1. `exercises.csv`

Houses instructional and solution texts for each problem.

| Column                | Description               |
| --------------------- | ------------------------- |
| `exercise_id`         | Unique problem identifier |
| `Instruction Content` | Problem description       |
| `Solution Content`    | Reference solution text   |

---

### 2. `submissions.csv`

Contains student submissions, their code, and timestamps.

| Column             | Description                            |
| ------------------ | -------------------------------------- |
| `x_user_id`        | Unique student identifier              |
| `exercise_id`      | Problem ID                             |
| `contents`         | Submitted code text                    |
| `created_datetime` | Submission time (Unix timestamp in ms) |

---

### 3. `questions.csv`

Logs of help center interactions (both student questions and teacher responses).

| Column                  | Description                                   |
| ----------------------- | --------------------------------------------- |
| `x_user_id`             | Student ID                                    |
| `exercise_id`           | Problem associated with the question          |
| `content`               | Text of the question or answer                |
| `post_created_datetime` | When the entry was posted                     |
| `is_student`            | `True` for student, `False` for teacher reply |

---

### 4. `scores.csv`

Label data indicating whether a student solved a problem correctly.

| Column        | Description           |
| ------------- | --------------------- |
| `x_user_id`   | Student ID            |
| `exercise_id` | Problem ID            |
| `label`       | Binary label (0 or 1) |

Note: These files must be properly aligned by `x_user_id` and `exercise_id`.

---

## Training

```bash
python main.py 
```

---

## Metrics

Evaluation is done using:

* Accuracy
* Precision
* Recall
* F1 Score
* AUC (ROC)

Sample output after each epoch:

```
Epoch 1/10 completed, Average Train Loss: 0.4593
Validation Metrics - Accuracy: 0.81, Precision: 0.79, Recall: 0.83, F1 Score: 0.81, AUC: 0.86
```

---

## Citation

If you use this code for your research, please cite appropriately. (Add your citation format here)

---

## Contact

For questions or feedback, feel free to contact the repository owner or open an issue.
xxxdokki@snu.ac.kr
