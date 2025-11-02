# Multi-Task-Learning-with-a-Two-Headed-MLP
Student Performance Multi-Task Learning (PyTorch)

This project implements a multi-task neural network using PyTorch to predict:
	1.	Final Grade (G3) — a regression task.
	2.	Romantic Relationship Status — a binary classification task.

The model jointly learns both tasks using a shared representation of student features, demonstrating how multi-task learning can improve generalization and data efficiency.

⸻

Dataset

Source: UCI Machine Learning Repository – Student Performance Dataset (ID 320)￼

The dataset contains attributes related to student demographics, social, and academic background.
Key features include:
	•	Demographic: sex, age, address, famsize, Pstatus
	•	Academic: studytime, failures, schoolsup, absences, G1, G2, G3
	•	Family & Social: Mjob, Fjob, guardian, internet, romantic

⸻

Data Pre-Processing
	1.	Feature / Target Separation
	•	G3 → Regression target (y_grade)
	•	romantic → Classification target (y_romantic)
	•	All other columns → Input features (X)
	2.	Encoding
	•	Binary columns (yes/no, F/M, etc.) mapped to 0/1
	•	Multi-category columns (Mjob, Fjob, guardian, etc.) one-hot encoded with pd.get_dummies()
	3.	Scaling
	•	Numerical columns standardized via StandardScaler (mean 0, std 1)
	4.	Splitting
	•	train_test_split → 70 % train / 15 % validation / 15 % test
	5.	Tensor Conversion & DataLoaders
	•	Custom StudentPerformanceDataset returning (X, y_grade, y_romantic)
	•	Separate DataLoaders for train, validation, and test sets.
