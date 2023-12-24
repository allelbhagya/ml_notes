Machine Learning Perspective of Data
Scales of Measurement
In the field of statistics and research, the measurement scale (also known as the level of measurement) refers to the nature of the values assigned to variables. There are four widely recognized scales of measurement: nominal, ordinal, interval, and ratio. Each scale has its own unique characteristics and properties.

1. **Nominal Scale:**
   - **Definition:** The nominal scale is the simplest level of measurement, where values are assigned to categories or labels without any inherent order or numerical significance. It only allows for the classification of data into distinct categories.
   - **Example:** Colors, gender, or types of fruit are examples of nominal variables. For instance, "red" and "blue" are labels, but there is no inherent order or numerical value associated with them.

2. **Ordinal Scale:**
   - **Definition:** The ordinal scale retains the property of classification like the nominal scale but also introduces the concept of order or ranking. However, the intervals between the ranks are not uniform, and the differences between the values are not meaningful.
   - **Example:** Educational levels (e.g., high school, bachelor's degree, master's degree) represent an ordinal scale. While we can rank them, the difference between high school and bachelor's degree does not necessarily represent a consistent or meaningful interval.

3. **Interval Scale:**
   - **Definition:** The interval scale includes both classification and ordered relationships, and it additionally ensures that the intervals between consecutive points are equal and meaningful. However, it lacks a true zero point, meaning that a value of zero does not indicate the absence of the measured quantity.
   - **Example:** Temperature measured in Celsius or Fahrenheit is an interval scale. The difference between 20 and 30 degrees is the same as the difference between 30 and 40 degrees, but a temperature of 0 degrees does not mean the absence of heat.

4. **Ratio Scale:**
   - **Definition:** The ratio scale is the most sophisticated level of measurement. It includes all the properties of the interval scale but also has a true zero point, indicating the complete absence of the measured quantity. Ratios of values are meaningful on this scale.
   - **Example:** Height, weight, income, and age (measured in years) are examples of ratio variables. For instance, a height of 0 inches or a weight of 0 pounds is meaningful, representing the absence of height or weight.

Dealing with Missing Data
Handling Categorical Data
Normalizing Data

Dealing with Missing Data:

1. **Removal of Missing Data:**
   - Remove rows or columns with missing values. This is suitable when the amount of missing data is small, and removing it doesn't significantly impact the dataset.

2. **Imputation:**
   - Fill in missing values with estimated or calculated values. Common imputation techniques include mean, median, or mode imputation for numerical data, or using the most frequent category for categorical data.
   
3. **Predictive Modeling:**
   - Use machine learning algorithms to predict missing values based on the observed data. This approach is effective when there is a pattern in the missing data that can be learned from the rest of the dataset.

4. **Multiple Imputation:**
   - Generate multiple plausible values for each missing data point, creating multiple complete datasets. This method accounts for the uncertainty associated with imputing missing values.

Handling Categorical Data:

1. **Label Encoding:**
   - Assign a unique numerical label to each category. This is useful when the categories have an ordinal relationship. However, it may introduce a false sense of order for nominal data.

2. **One-Hot Encoding:**
   - Create binary columns for each category, representing the presence or absence of that category. This is suitable for nominal data and prevents the model from assuming false ordinal relationships.

3. **Ordinal Encoding:**
   - Assign numerical values to categories based on their inherent order. This is appropriate when there is a meaningful order among the categories.

4. **Binary Encoding:**
   - Represent each category with binary code. This reduces the dimensionality compared to one-hot encoding while preserving information about the categories.

Normalizing Data:

1. **Min-Max Scaling (Normalization):**
   - Scale numerical features to a specific range (e.g., 0 to 1) by subtracting the minimum value and dividing by the range. This is sensitive to outliers but ensures that all features have the same scale.

2. **Z-score Standardization:**
   - Transform data to have a mean of 0 and a standard deviation of 1. This method is less sensitive to outliers but maintains the shape of the distribution.

3. **Robust Scaling:**
   - Scale data based on the median and interquartile range, making it robust to outliers. It is suitable when the data contains extreme values.

4. **Log Transformation:**
   - Apply a logarithmic function to the data to reduce the impact of skewed distributions and make the data more normally distributed.

**Feature Construction or Generation:**

Feature construction or generation refers to the process of creating new features from the existing ones in a dataset. This can help improve the performance of machine learning models by providing them with more relevant and informative input variables. Here are some techniques for feature construction or generation:

1. **Polynomial Features:**
   - Create new features by raising existing features to a higher power. This is particularly useful when there are nonlinear relationships between the features and the target variable.

2. **Interaction Terms:**
   - Combine two or more existing features to create new features that capture interactions between them. This is valuable when the combined effect of features is significant.

3. **Binning or Discretization:**
   - Convert continuous features into discrete bins or categories. This can help capture non-linear relationships and make the model more robust to outliers.

4. **Feature Scaling:**
   - Scale features to a standard range. This is not exactly creating new features, but it can be considered a form of feature transformation that can improve the model's performance.

5. **Feature Crosses:**
   - Combine two or more categorical features to create new features. This is especially useful when there are interactions between different categorical variables.

6. **Embedding:**
   - Represent categorical variables as continuous vectors in a lower-dimensional space. This is often used in natural language processing (NLP) for word embeddings.

7. **Domain-Specific Feature Engineering:**
   - Leverage domain knowledge to create new features that are relevant to the problem at hand. For example, in a time-series analysis, creating lag features might be beneficial.

**Correlation and Causation:**

**Correlation:**
   - **Definition:** Correlation measures the statistical association between two variables. It quantifies the degree to which changes in one variable correspond to changes in another. The correlation coefficient ranges from -1 to 1, where -1 indicates a perfect negative correlation, 1 indicates a perfect positive correlation, and 0 indicates no correlation.
   - **Caution:** Correlation does not imply causation. Two variables may be correlated, but it doesn't mean that one causes the other. There could be confounding factors or the correlation might be coincidental.

**Causation:**
   - **Definition:** Causation implies a cause-and-effect relationship between two variables. Establishing causation is more complex than identifying correlation. Experimental design, controlled studies, and statistical methods such as randomized controlled trials (RCTs) are often used to infer causation.
   - **Correlation vs. Causation:** Correlation is a necessary but not sufficient condition for causation. While causation implies correlation, correlation does not necessarily imply causation.

**Polynomial Regression:**

**Definition:**
Polynomial regression is a type of regression analysis where the relationship between the independent variable \(x\) and the dependent variable \(y\) is modeled as an \(n\)-th degree polynomial. While simple linear regression assumes a linear relationship, polynomial regression can capture more complex, non-linear patterns in the data.

**Mathematical Representation:**
The polynomial regression equation of degree \(n\) is given by:
\[ y = \beta_0 + \beta_1x + \beta_2x^2 + \ldots + \beta_nx^n + \varepsilon \]
where:
- \(y\) is the dependent variable,
- \(x\) is the independent variable,
- \(n\) is the degree of the polynomial,
- \(\beta_0, \beta_1, \ldots, \beta_n\) are the coefficients,
- \(\varepsilon\) is the error term.

**Use Cases:**
Polynomial regression is useful when the relationship between the variables is not a straight line. It is applied in scenarios where the data exhibits curvature, peaks, or troughs, and a linear model would not accurately represent the underlying pattern.

**Logistic Regression:**

**Definition:**
Logistic regression is a statistical model used for binary classification. Despite its name, logistic regression is employed for classification, not regression. It models the probability that a given instance belongs to a particular class.

**Mathematical Representation:**
The logistic regression model for binary classification is expressed using the logistic function (sigmoid function):
\[ P(Y=1) = \frac{1}{1 + e^{-(\beta_0 + \beta_1x_1 + \beta_2x_2 + \ldots + \beta_nx_n)}} \]
where:
- \( P(Y=1) \) is the probability of the positive class,
- \( x_1, x_2, \ldots, x_n \) are the features,
- \( \beta_0, \beta_1, \ldots, \beta_n \) are the coefficients,
- \( e \) is the base of the natural logarithm.

**Use Cases:**
Logistic regression is commonly used when the dependent variable is binary (two classes). It's widely applied in fields such as medicine (disease diagnosis), finance (credit scoring), and various other areas where binary classification is required.

In summary, polynomial regression is used for modeling non-linear relationships in regression tasks, while logistic regression is employed for binary classification tasks, modeling the probability of belonging to a particular class.

ROC Curve
![image](https://github.com/allelbhagya/ml_notes/assets/80905783/7c3ef0f8-5b4a-41be-abd0-5f6a76f2aed8)


The AUC-ROC curve, or Area Under the Receiver Operating Characteristic curve, is a graphical representation of the performance of a binary classification model at various classification thresholds. It is commonly used in machine learning to assess the ability of a model to distinguish between two classes, typically the positive class (e.g., presence of a disease) and the negative class (e.g., absence of a disease).

ROC stands for Receiver Operating Characteristics, and the ROC curve is the graphical representation of the effectiveness of the binary classification model. It plots the true positive rate (TPR) vs the false positive rate (FPR) at different classification thresholds.


AUC Curve:
AUC stands for Area Under the Curve, and the AUC curve represents the area under the ROC curve. It measures the overall performance of the binary classification model. As both TPR and FPR range between 0 to 1, So, the area will always lie between 0 and 1, and A greater value of AUC denotes better model performance. Our main goal is to maximize this area in order to have the highest TPR and lowest FPR at the given threshold. The AUC measures the probability that the model will assign a randomly chosen positive instance a higher predicted probability compared to a randomly chosen negative instance.

 It represents the probability with with our model is able to distinguish between the two classes which are present in our target. 


TPR and FPR
This is the most common definition that you would have encountered when you would Google AUC-ROC. Basically, the ROC curve is a graph that shows the performance of a classification model at all possible thresholds( threshold is a particular value beyond which you say a point belongs to a particular class). The curve is plotted between two parameters

TPR – True Positive Rate
FPR – False Positive Rate
Before understanding, TPR and FPR let us quickly look at the confusion matrix.

Confusion Matrix for a Classification Task
Confusion Matrix for a Classification Task

True Positive: Actual Positive and Predicted as Positive
True Negative: Actual Negative and Predicted as Negative
False Positive(Type I Error): Actual Negative but predicted as Positive
False Negative(Type II Error): Actual Positive but predicted as Negative

**Sensitivity / True Positive Rate / Recall (TPR):**
- **Definition:** Sensitivity, True Positive Rate (TPR), or Recall measures the ability of a classification model to correctly identify positive instances out of all actual positive instances.
- **Formula:** \[ TPR = \frac{TP}{TP + FN} \]
  - \(TP\) (True Positive): Instances that are actually positive and are correctly identified as positive.
  - \(FN\) (False Negative): Instances that are actually positive but are incorrectly identified as negative.

**False Positive Rate (FPR):**
- **Definition:** False Positive Rate (FPR) measures the ratio of negative instances that are incorrectly classified as positive by the model.
- **Formula:** \[ FPR = \frac{FP}{TN + FP} \]
  - \(FP\) (False Positive): Instances that are actually negative but are incorrectly identified as positive.
  - \(TN\) (True Negative): Instances that are actually negative and are correctly identified as negative.

**Relation between FPR and Specificity:**
- \[ FPR = 1 - \text{Specificity} \]
  - Specificity is another performance metric that measures the ability of a model to correctly identify negative instances out of all actual negative instances. The formula for Specificity is \(\frac{TN}{TN + FP}\).
  - Since FPR is complementary to Specificity, you can express FPR as \(1 - \text{Specificity}\).

In summary, Sensitivity focuses on the positive class and measures the model's ability to capture all actual positive instances. FPR, on the other hand, assesses the model's tendency to misclassify negative instances as positive. These metrics are essential in evaluating the performance of binary classification models, particularly when the class distribution is imbalanced.
Specificity measures the proportion of actual negative instances that are correctly identified by the model as negative. It represents the ability of the model to correctly identify negative instances
