## Introduction: History and Evolution

A subset of artificial intelligence known as machine learning focuses primarily on the creation of algorithms that enable a computer to independently learn from data and previous experiences. Arthur Samuel first used the term "machine learning" in 1959. It could be summarized as follows:

Without being explicitly programmed, machine learning enables a machine to automatically learn from data, improve performance from experiences, and predict things.

Machine learning algorithms create a mathematical model that, without being explicitly programmed, aids in making predictions or decisions with the assistance of sample historical data, or training data. For the purpose of developing predictive models, machine learning brings together statistics and computer science. Algorithms that learn from historical data are either constructed or utilized in machine learning. The performance will rise in proportion to the quantity of information we provide.

A machine can learn if it can gain more data to improve its performance.

The early history of Machine Learning (Pre-1940):
1834: In 1834, Charles Babbage, the father of the computer, conceived a device that could be programmed with punch cards. However, the machine was never built, but all modern computers rely on its logical structure.
1936: In 1936, Alan Turing gave a theory that how a machine can determine and execute a set of instructions.
The era of stored program computers:
1940: In 1940, the first manually operated computer, "ENIAC" was invented, which was the first electronic general-purpose computer. After that stored program computer such as EDSAC in 1949 and EDVAC in 1951 were invented.
1943: In 1943, a human neural network was modeled with an electrical circuit. In 1950, the scientists started applying their idea to work and analyzed how human neurons might work.
Computer machinery and intelligence:
1950: In 1950, Alan Turing published a seminal paper, "Computer Machinery and Intelligence," on the topic of artificial intelligence. In his paper, he asked, "Can machines think?"
Machine intelligence in Games:
1952: Arthur Samuel, who was the pioneer of machine learning, created a program that helped an IBM computer to play a checkers game. It performed better more it played.
1959: In 1959, the term "Machine Learning" was first coined by Arthur Samuel.
The first "AI" winter:
The duration of 1974 to 1980 was the tough time for AI and ML researchers, and this duration was called as AI winter.
In this duration, failure of machine translation occurred, and people had reduced their interest from AI, which led to reduced funding by the government to the researches.
Machine Learning from theory to reality
1959: In 1959, the first neural network was applied to a real-world problem to remove echoes over phone lines using an adaptive filter.
1985: In 1985, Terry Sejnowski and Charles Rosenberg invented a neural network NETtalk, which was able to teach itself how to correctly pronounce 20,000 words in one week.
1997: The IBM's Deep blue intelligent computer won the chess game against the chess expert Garry Kasparov, and it became the first computer which had beaten a human chess expert.
Machine Learning at 21st century
2006:

Geoffrey Hinton and his group presented the idea of profound getting the hang of utilizing profound conviction organizations.
The Elastic Compute Cloud (EC2) was launched by Amazon to provide scalable computing resources that made it easier to create and implement machine learning models.
2007:

Participants were tasked with increasing the accuracy of Netflix's recommendation algorithm when the Netflix Prize competition began.
Support learning made critical progress when a group of specialists utilized it to prepare a PC to play backgammon at a top-notch level.

2008:

Google delivered the Google Forecast Programming interface, a cloud-based help that permitted designers to integrate AI into their applications.
Confined Boltzmann Machines (RBMs), a kind of generative brain organization, acquired consideration for their capacity to demonstrate complex information conveyances.
2009:

Profound learning gained ground as analysts showed its viability in different errands, including discourse acknowledgment and picture grouping.
The expression "Large Information" acquired ubiquity, featuring the difficulties and open doors related with taking care of huge datasets.
2010:

The ImageNet Huge Scope Visual Acknowledgment Challenge (ILSVRC) was presented, driving progressions in PC vision, and prompting the advancement of profound convolutional brain organizations (CNNs).
2011:

On Jeopardy! IBM's Watson defeated human champions., demonstrating the potential of question-answering systems and natural language processing.
2012:

AlexNet, a profound CNN created by Alex Krizhevsky, won the ILSVRC, fundamentally further developing picture order precision and laying out profound advancing as a predominant methodology in PC vision.
Google's Cerebrum project, drove by Andrew Ng and Jeff Dignitary, utilized profound figuring out how to prepare a brain organization to perceive felines from unlabeled YouTube recordings.
2013:

Ian Goodfellow introduced generative adversarial networks (GANs), which made it possible to create realistic synthetic data.
Google later acquired the startup DeepMind Technologies, which focused on deep learning and artificial intelligence.
2014:

Facebook presented the DeepFace framework, which accomplished close human precision in facial acknowledgment.
AlphaGo, a program created by DeepMind at Google, defeated a world champion Go player and demonstrated the potential of reinforcement learning in challenging games.
2015:

Microsoft delivered the Mental Toolbox (previously known as CNTK), an open-source profound learning library.
The performance of sequence-to-sequence models in tasks like machine translation was enhanced by the introduction of the idea of attention mechanisms.
2016:

The goal of explainable AI, which focuses on making machine learning models easier to understand, received some attention.
Google's DeepMind created AlphaGo Zero, which accomplished godlike Go abilities to play without human information, utilizing just support learning.
2017:

Move learning acquired noticeable quality, permitting pretrained models to be utilized for different errands with restricted information.
Better synthesis and generation of complex data were made possible by the introduction of generative models like variational autoencoders (VAEs) and Wasserstein GANs.


## Machine Learning Categories: Supervised Learning, Unsupervised Learning, Reinforcement Learning
1. **Supervised Learning:**
   - **Definition:** In supervised learning, the algorithm is trained on a labeled dataset, where each input is associated with the corresponding correct output. The algorithm learns to map the input data to the correct output by generalizing from the labeled examples.
   - **Example:** Consider a spam email filter as an example. The algorithm is trained on a dataset of emails labeled as either spam or not spam. It learns to distinguish between the two based on features such as keywords, sender information, and email structure.

2. **Unsupervised Learning:**
   - **Definition:** Unsupervised learning deals with unlabeled data, where the algorithm explores the inherent structure or patterns within the data without explicit guidance. The goal is often to discover hidden relationships or groupings within the dataset.
   - **Example:** Clustering is a common unsupervised learning task. Imagine a dataset of customer purchase histories. The algorithm might identify natural groupings of similar purchasing behavior without any prior information about which group corresponds to which type of customer.

3. **Reinforcement Learning:**
   - **Definition:** Reinforcement learning involves an agent that learns to make decisions by interacting with an environment. The agent receives feedback in the form of rewards or penalties based on the actions it takes. The objective is for the agent to learn a strategy (policy) that maximizes the cumulative reward over time.
   - **Example:** Think of a self-learning game-playing AI, like a computer program that plays chess. The AI makes moves, receives feedback (win, lose, or draw), and adjusts its strategy over time to improve its chances of winning future games.


## Knowledge Discovery in Databases - (CIST MER)

KDD Process
KDD (Knowledge Discovery in Databases) is a process that involves the extraction of useful, previously unknown, and potentially valuable information from large datasets. The KDD process is an iterative process and it requires multiple iterations of the above steps to extract accurate knowledge from the data.The following steps are included in KDD process: 

1. Data Cleaning
Data cleaning is defined as removal of noisy and irrelevant data from collection. 

Cleaning in case of Missing values.
Cleaning noisy data, where noise is a random or variance error.
Cleaning with Data discrepancy detection and Data transformation tools.
2. Data Integration
Data integration is defined as heterogeneous data from multiple sources combined in a common source(DataWarehouse). Data integration using Data Migration tools, Data Synchronization tools and ETL(Extract-Load-Transformation) process.

3. Data Selection
Data selection is defined as the process where data relevant to the analysis is decided and retrieved from the data collection. For this we can use  Neural network, Decision Trees, Naive bayes, Clustering, and Regression methods. 

4. Data Transformation
Data Transformation is defined as the process of transforming data into appropriate form required by mining procedure. Data Transformation is a two step process: 

5. Data Mapping: Assigning elements from source base to destination to capture transformations.
Code generation: Creation of the actual transformation program.
Data Mining
Data mining is defined as techniques that are applied to extract patterns potentially useful. It transforms task relevant data into patterns, and decides purpose of model using classification or characterization.

6. Pattern Evaluation
Pattern Evaluation is defined as identifying strictly increasing patterns representing knowledge based on given measures. It find interestingness score of each pattern, and uses summarization and Visualization to make data understandable by user.

7. Knowledge Representation
This involves presenting the results in a way that is meaningful and can be used to make decisions.

KDD process

![KDD_process](https://github.com/allelbhagya/ml_notes/assets/80905783/a003320a-3cfc-44a4-8975-814898705dd3)

Advantages of KDD
Improves decision-making: KDD provides valuable insights and knowledge that can help organizations make better decisions.
Increased efficiency: KDD automates repetitive and time-consuming tasks and makes the data ready for analysis, which saves time and money.
Better customer service: KDD helps organizations gain a better understanding of their customers’ needs and preferences, which can help them provide better customer service.
Fraud detection: KDD can be used to detect fraudulent activities by identifying patterns and anomalies in the data that may indicate fraud.
Predictive modeling: KDD can be used to build predictive models that can forecast future trends and patterns.
Disadvantages of KDD
Privacy concerns: KDD can raise privacy concerns as it involves collecting and analyzing large amounts of data, which can include sensitive information about individuals.
Complexity: KDD can be a complex process that requires specialized skills and knowledge to implement and interpret the results.
Unintended consequences: KDD can lead to unintended consequences, such as bias or discrimination, if the data or models are not properly understood or used.
Data Quality: KDD process heavily depends on the quality of data, if data is not accurate or consistent, the results can be misleading
High cost: KDD can be an expensive process, requiring significant investments in hardware, software, and personnel.
Overfitting: KDD process can lead to overfitting, which is a common problem in machine learning where a model learns the detail and noise in the training data to the extent that it negatively impacts the performance of the model on new unseen data. 


SEMMA (Sample, Explore, Modify, Model, Assess).

SEMMA is the sequential methods to build machine learning models incorporated in ‘SAS Enterprise Miner’, a product by SAS Institute Inc., one of the largest producers of commercial statistical and business intelligence software. However, the sequential steps guide the development of a machine learning system. Let’s look at the five sequential steps to understand it better.


## SEMMA model in Machine Learning

Sample: This step is all about selecting the subset of the right volume dataset from a large dataset provided for building the model. It will help us to build the model very efficiently. Basically in this step, we identify the independent variables(outcome) and dependent variables(factors). The selected subset of data should be actually a representation of the entire dataset originally collected, which means it should contain sufficient information to retrieve. The data is also divided into training and validation purpose.

Explore: In this phase, activities are carried out to understand the data gaps and relationship with each other. Two key activities are univariate and multivariate analysis. In univariate analysis, each variable looks individually to understand its distribution, whereas in multivariate analysis the relationship between each variable is explored. Data visualization is heavily used to help understand the data better. In this step, we do analysis with all the factors which influence our outcome.

Modify: In this phase, variables are cleaned where required. New derived features are created by applying business logic to existing features based on the requirement. Variables are transformed if necessary. The outcome of this phase is a clean dataset that can be passed to the machine learning algorithm to build the model. In this step, we check whether the data is completely transformed or not. If we need the transformation of data we use the label encoder or label binarizer.

Model: In this phase, various modelling or data mining techniques are applied to the pre-processed data to benchmark their performance against desired outcomes. In this step, we perform all the mathematical which makes our outcome more precise and accurate as well.

Assess: This is the last phase. Here model performance is evaluated against the test data (not used in model training) to ensure reliability and business usefulness. Finally, in this step, we perform the evaluation and interpretation of data. We compare our model outcome with the actual outcome and analysis of our model limitation and also try to overcome that limitation.

![image](https://github.com/allelbhagya/ml_notes/assets/80905783/3c4907d2-b89a-45df-8335-0808e2986507)
