##### MOTIVATION FOR CHOOSING THE DATA SCIENCE:
    
A lot of people are inclined towards __Data Science__ these days and so am I. However, my reason is different. I am actually a___Mathematician___ by brain. In addition, I chose ___Computer Science___ as my __Profession__ which is a general field consisting of a lot of sub-emerging fields:

* Data Science
* Cyber Security
* IT
* Game Development
* App Development
* Web Development
* And etc

Everyone is in a race of chosing the best profession in their respective ways. For me, I chose the Data Science Profession because this is the one profession that seems to be my own. In short, my interests and abilities incline in the favour of this Field of Data Science.

##### Motivation Of Doing Data Analysis On This Specific DataSet:

These days many companies, firms, and organiztions are working on increasing the customer satisfaction level. An important factor that plays a decisive role in this regards is to get the sentiment of the customer / client including what he/she wants to ask, what is the customer's / client's tone, etc. Without knowledge of the customer's / client's sentiment, it becomes very difficult for any company, firm, organization, etc to answer correctly.

To solve this problem, a very fine method that can be applied is to make a prediction model. That's what I did. I made two Prediction Models with different accuracies. They can predict the sentiment of a person who posted something on Twitter based on the DataSet: https://www.kaggle.com/datasets/nursyahrina/chat-sentiment-dataset (URL).

##### Complete Description About The Project Work-Flow

The DataSet  used for __Data Analysis__ here is extracted from __Twitter__ and is present at this _URL_ : https://www.kaggle.com/datasets/nursyahrina/chat-sentiment-dataset .

This DataSet Contained 4 Attributes (i.e Columns):

* Id
* Sentiment
* Tool
* Text

##### Problems I Faced:

1. Column Names were undefined and very Ambiguous
2. After Preprocessing, the DataFrame extends to an extent that it is very difficult to save that DataFrame into a file for later use becuase it consumes more than 13 GB of RAM for this task
3. There were not much attributes in the DataSet to Visualize from different angles

##### Solutions:

1. Created copies of old columns with better defining names and removed the old columns
2. Instead of saving the DataFrame, save the models including OneHotEncoder, LabelEncoder, Scaler and Vectorizer.
3. Visualized The Cleaned DataSet using 3 Graphs:
   1. Pie Chart (Overall Distribution)
   2. Bar Chart (Overall Distribution)
   3. CountPlot (Detailed Overall Distribution)

##### EDA Steps:

- Step 1: (Information About DataSet)

  - Extracted:

    - The structure
    - Information about datatypes
    - Count of nulls in each column of DataSet
- Step 2: (Data Cleaning)

  - Removed the records with any null values
  - Removed the duplicate records
- Step 3: (Data Visualization)

  - Used the Cleaned Data To Visualize Data using:
    - Pie Chart
    - Bar Chart
    - Count Plot
- Step 4: (Data Encoding/Preprocessing)

  - Used Encoders (e.g. Label Encoder, One Hot Encoder) for encoding str-type columns
  - Used Scaler to scale numeric features based on a standard
  - Used Vectorizer (TFIDF-Vectorizer) to Vectorize the column to be predicted
  - Saved Models instead of updated Data (using joblib)
- Step 5: (Training The Models)

  - Loaded preprocessing models to get Preprocessed Data
  - Trained Two Models based on the Preprocessed Data
  - Evulated Accuracy and Compared
  - Saved The Models (using joblib)
- Step 6: (Using The Models)

  - Load The Models Back (using joblib)
  - Get Input from user and convert it to model-readable format
  - Get Predictions using Both Models

##### Tools Used:
- Visual Studio Code
- Jupyter NoteBook
- Python Libraries
    - Visualization: matplotlib.pyplot, seaborn
    - Data Cleaning and Manipulation: pandas, numpy(a litle bit)
    - Models: sklearn for encoders, scalers, and prediction models
- git and GitHub

Requirements to Run the files:
- python 3.13.5 (Recommended) {If you are using an older version of ___Python___, this code may not run properly in some cases}
- pandas
- numpy
- scikit-learn
- warnings (Not Necessary but Recommended)
- matplotlib
- seaborn

For Ease of Reading and Reviews:
    - The Actual Code Files' names start with step.1_file_name.py where step is the step # same as discussed above
    - The NoteBook Files' names start with step.2_file_name.ipynb where step is the step # same as discussed above
    - For the same step the Actual Code File and NoteBook File just differ in their extension (i.e .py, .ipynb) and their sub_branch #, for example:
        - Actual Code File: 1.1_Data_Info.py
        - NoteBook File: 1.2_Data_Info.ipynb
    - For the same step the Actual Code File and Jupyter NoteBook File are placed in the same folder whose name is the same as the title of the respective step.
