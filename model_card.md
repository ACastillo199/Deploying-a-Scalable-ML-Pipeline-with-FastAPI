# Model Card

For additional information see the Model Card paper: [https://arxiv.org/pdf/1810.03993.pdf](https://arxiv.org/pdf/1810.03993.pdf)

## Model Details
This model was developed as part of a scalable machine learning pipeline project. It utilizes a Random Forest Classifier (`sklearn.ensemble.RandomForestClassifier`) version 1.0. All trained model artifacts and preprocessors are serialized and stored in the `model/` directory, including `model.pkl`, `encoder.pkl`, and `lb.pkl`.

## Intended Use
This model is primarily intended to predict whether an individual's income exceeds $50,000 per year based on U.S. Census data attributes such as education, age, occupation, and hours worked. Primary intended users include educational users, data scientists, and researchers analyzing demographic and economic classification models. This model should not be used for high-stakes automated financial decisions, credit scoring, or employment screening without rigorous bias auditing and legal compliance checks.

## Training Data
The source data used for training is derived from the 1994 U.S. Census Bureau database (`census.csv`). The data was split into an 80% training dataset and a 20% test dataset using stratified splitting based on the target label. Categorical features were processed using `OneHotEncoder` with unknown categories ignored, and target labels were binarized using `LabelBinarizer`.

## Evaluation Data
The model was evaluated on a held-out 20% test split from the census dataset comprising thousands of unseen samples. Performance was additionally evaluated across specific categorical feature slices—such as subsets grouped by sex, race, education, and occupation—to verify robustness and identify potential performance disparities.

## Metrics
The performance metrics used to evaluate the model are Precision, Recall, and F1-Score (computed as an FBeta score with $\beta=1$). The overall model performance on the test set yielded a Precision of approximately 0.7353, a Recall of approximately 0.6378, and an F1-Score of approximately 0.6831. Detailed performance metrics across individual categorical slices have been computed and exported to `slice_output.txt`.

## Ethical Considerations
The dataset includes demographic features such as race, sex, and native country. Machine learning models trained on socioeconomic data can inadvertently perpetuate historical biases or wage disparities. To mitigate this risk, performance across demographic slices found in `slice_output.txt` should be carefully reviewed to ensure the model does not exhibit extreme performance gaps or discriminatory behavior against underrepresented groups.

## Caveats and Recommendations
The underlying dataset is based on historical 1994 census data and may not accurately reflect modern economic conditions, inflation rates, wage structures, or demographic distributions. Users should treat predictions as exploratory estimates rather than absolute ground truths. Future iterations of this pipeline should incorporate updated demographic datasets and hyperparameter tuning to optimize recall and overall fairness.