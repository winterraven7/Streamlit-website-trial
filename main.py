import streamlit as st
import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import TPESampler
import pickle
from IPython.core.display import HTML
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
import seaborn as sns
from scipy.stats import norm

# App layout
# Initialize session_state to track the selected project
if 'selected_project' not in st.session_state:
    st.session_state.selected_project = None


# Function to handle project selection
def select_project(project_name):
    st.session_state.selected_project = project_name


st.sidebar.image("https://github.com/winterraven7/Streamlit-website-trial/blob/main/logo.jpg", use_column_width=True)
st.sidebar.button("Homepage", on_click=select_project, args=("Homepage",))
st.sidebar.title("Projects")
# Create buttons for each project, binding them to the same function but with different arguments
st.sidebar.button("Project 1: Steel Plate Defect Detection", on_click=select_project, args=("Project 1",))
st.sidebar.button("Game: Predict the Admission Probability", on_click=select_project, args=("Game",))
st.sidebar.button("Project 3", on_click=select_project, args=("Project 3",))


def home_page():
    st.title("Hello! Welcome to my data science pocket :sparkles: ")
    st.subheader("**Self Introduction**")
    st.write("Hey there! I'm Yian Zhang, a passionate Data Science learner dedicated to turning complex data into "
             "actionable"
             "insights. Welcome to my personal space where I share my journey, projects, and the occasional "
             "coffee-fueled late-night coding adventure.")

    st.markdown('### About Me :coffee:')
    st.write("Throughout my academic career, I’ve been driven by a curiosity to understand how data can be used to "
             "make informed decisions and drive change. Holding an undergraduate degree in economics, I've acquired a "
             "robust foundation for integrating and applying data insights to real-world scenarios.  I’ve excelled in "
             "my coursework and plan to actively participate in data science competitions and hackathons in the "
             "future, sharpening my skills.")

    st.markdown('### My Expertise :seedling:')
    st.write("My toolbox includes Python, R, SQL, TensorFlow, and PyTorch, among others. I specialize in machine "
             "learning, statistical analysis, and data visualization, and I'm learning new techniques and "
             "technologies to follow the cutting edge of data science.")

    # Placeholder for chatbot - you can integrate an actual chatbot here
    # Chatbot section
    st.subheader("Ask a Question")

    # Define a dictionary of questions and answers
    qa_pairs = {
        "What is Project 1 about?": "Project 1 is about...",
        "How can I contribute to Project 2?": "You can contribute to Project 2 by...",
        "What are the goals of Project 3?": "The goals of Project 3 include..."
    }

    # Use a selectbox for the questions
    selected_question = st.selectbox("Choose your question", options=list(qa_pairs.keys()))

    # Display the answer to the selected question
    if st.button("Get Answer"):
        st.write(qa_pairs[selected_question])


# Placeholder for steel plate defect project
@st.cache
def project1_steel_plate_defect():
    st.header("Project 1: Steel Plate Defect Detection")
    st.write("This section will demonstrate the analysis and prediction model for steel plate defect detection."
             "\n Key words: EDA, XGBoost.")
    st.write("Massaron, Luca. \"Steel Plate EDA & XGBoost is All You Need.\" Kaggle. "
             "https://www.kaggle.com/code/lucamassaron/steel-plate-eda-xgboost-is-all-you-need."
             )
    url1 = 'https://github.com/winterraven7/Streamlit-website-trial/blob/main/train.csv'
    url2 = 'https://github.com/winterraven7/Streamlit-website-trial/blob/main/test.csv'
    train = pd.read_csv(url1)
    test = pd.read_csv(url2)

    st.subheader("**Load Data**")

    st.write("The shape of training set", train.shape)
    st.write("The shape of testing set", test.shape)
    st.write("There are 34 variables in train dataset but only 27 in test dataset.")

    st.write("Data overview of training set:", train.head())

    st.subheader("**Brief Exploratory of Data**")
    st.write("The basic description of training set", train.describe().T)

    st.write("Check how many nulls are our your dataset and calculate the ratio of nulls for each variable."
             " It turns out that all cells are filled.")
    nonnull_counts = train.notnull().sum()
    null_counts = train.isnull().sum()
    total = nonnull_counts + null_counts
    nonnull_percentage = nonnull_counts / total
    null_percentage = null_counts / total

    fig, ax = plt.subplots(figsize=(10, 5))  # Adjust the size as needed
    ax.bar(nonnull_percentage.index, nonnull_percentage, label='Filled', color='lightseagreen')
    ax.bar(null_percentage.index, null_percentage, bottom=nonnull_percentage, label='Null', color='red')
    ax.set_ylabel('Percentage of filled cells')
    ax.set_title('Percentage of filled cells per table')
    plt.xticks(rotation=90)  # Rotate the x-axis labels if necessary
    ax.legend()

    st.pyplot(fig)

    st.write("Check the repeatability of observation.")
    train_duplicates_number = train[train.duplicated()]
    test_duplicates_number = test[test.duplicated()]

    st.write("train_duplicates_number", len(train_duplicates_number))
    st.write("test_duplicates_number", len(test_duplicates_number))

    X = train.drop(['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains',
                    'Dirtiness', 'Bumps', 'Other_Faults'], axis=1)

    y = train[['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']]

    st.write("Next is One-hot coding. We will split X and Y.")
    code = """
    X = train.drop(['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains',
                    'Dirtiness', 'Bumps', 'Other_Faults'], axis=1)

    y = train[['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']]
    """

    st.code(code, language='python')

    st.markdown("""
    #### Data Distribution Observations

    1. **Skewed Distributions**: Many variables like `X_Minimum`, `Y_Minimum`, `Pixels_Areas`, etc., show a right-skewed distribution, meaning most values are clustered towards the left with a long tail extending to the right.

    2. **Outliers**: Some variables such as `Pixels_Areas`, `X_Perimeter`, and `TypeOfSteel_A300` have outliers, indicated by the bars far away from the cluster of other bars.

    3. **Bimodal/Multimodal Distributions**: Variables like `TypeOfSteel_A400` exhibit bimodal or multimodal distributions, suggesting multiple peaks and possibly subgroups within the data.

    4. **Uniform Distributions**: Some variables, e.g., `Outside_Global_Index`, appear to have a uniform distribution with values occurring with similar frequency.

    5. **Narrow Distributions**: Variables such as `Edges_X_Index`, `Edges_Y_Index`, and `Outside_X_Index` show narrow distributions, indicating that the values are concentrated in a small range.

    6. **Sparse Data**: Certain variables like `Pastry`, `Z_Scratch`, `K_Scatch`, `Stains`, and `Other_Faults` are sparse, with most of their values being zero or close to zero.

    7. **Differing Ranges**: The scales of the variables vary significantly, for instance, `Length_Of_Conveyor` values are in the tens of thousands, whereas `Luminosity_Index` values are close to zero, indicating different units or scales.
    """)
    sns.set(rc={'figure.figsize': (20, 16)})
    fig, ax = plt.subplots()
    train.hist(color='c', ax=ax)
    st.pyplot(fig)

    st.subheader("**Feature Engineering**")
    st.write("By calculating the range of coordinates (both in the x and y directions), we capture information about "
             "the"
             "spatial extent of each fault. This could be relevant because faults with larger spatial extents might "
             "indicate more severe defects or anomalies in the steel plate."
             " We will check whether the distributions of training set and testing set are identical or not.")

    def calculate_coordinate_range_features(data):
        data['X_Range'] = (data['X_Maximum'] - data['X_Minimum'])
        data['Y_Range'] = (data['Y_Maximum'] - data['Y_Minimum'])
        return data

    train = calculate_coordinate_range_features(train)
    test = calculate_coordinate_range_features(test)

    code = """
    def calculate_coordinate_range_features(data):
    data['X_Range'] = (data['X_Maximum'] - data['X_Minimum'])
    data['Y_Range'] =( data['Y_Maximum'] - data['Y_Minimum'])
    return data
    
    train = calculate_coordinate_range_features(train)
    test = calculate_coordinate_range_features(test)
    """
    code = """
    def plot_distribution_pairs(train, test, feature, hue="set", palette=None):
        data_df = train.copy()
        data_df['set'] = 'train'
        data_df = pd.concat([data_df, test.copy()]).fillna('test')
        data_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        f, axes = plt.subplots(1, 2, figsize=(14, 6))
        for i, s in enumerate(data_df[hue].unique()):
            selection = data_df.loc[data_df[hue] == s, feature]
            q_025, q_975 = np.percentile(selection, [2.5, 97.5])
            selection_filtered = selection[(selection >= q_025) & (selection <= q_975)]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                sns.histplot(selection_filtered, color=palette[i], ax=axes[0], label=s)
                sns.boxplot(x=hue, y=feature, data=data_df, palette=palette, ax=axes[1])
        axes[0].set_title(f"Paired train/test distributions of {feature}")
        axes[1].set_title(f"Paired train/test boxplots of {feature}")
        axes[0].legend()
        axes[1].legend()
        plt.tight_layout()
        return f
        
    train = calculate_coordinate_range_features(train)
    test = calculate_coordinate_range_features(test)
    """
    st.code(code, language='python')

    st.image("https://github.com/winterraven7/Streamlit-website-trial/blob/main/x-range.jpg", use_column_width=True)
    st.image("https://github.com/winterraven7/Streamlit-website-trial/blob/main/y-range.jpg", use_column_width=True)


def project2_prediction_uni_admission():
    st.header("Game: Calculate the Admission Probability")
    st.write("This \"prediction\" is just fun and there is no serious theorem behind it. Taking the non-seriousness "
             "and uncertainty in real world into account，we will use normal dictribution here.")

    def calculate_probability(a, b, c):
        a = a/4
        b = b/120
        c = c/100
        x = 0.4 * a + 0.4 * b + 0.2 * c
        probability = norm.cdf(x)
        return probability

    # Input fields for GPA, TOEFL score, and QS Rank of University
    gpa = st.number_input('Enter your GPA:', min_value=0.0, max_value=4.0, value=0.0, format="%.2f")
    toefl_score = st.number_input('Enter your TOEFL Score:', min_value=0, max_value=120, value=0, step=1)
    qs_rank = st.number_input('Enter QS Rank of your University:', min_value=1, value=1)

    # Calculate button
    if st.button('Calculate Probability'):
        probability = calculate_probability(gpa, toefl_score, qs_rank)
        st.write(f"The Probability: {probability:.4f}")


# Change content based on the selected project
# Check if 'selected_project' is not in session_state or it's set to "Homepage"
if st.session_state.selected_project is None or st.session_state.selected_project == "Homepage":
    home_page()  # Call the function to render the homepage
elif st.session_state.selected_project == "Project 1":
    project1_steel_plate_defect()  # Call the function for Project 1
elif st.session_state.selected_project == "Game":
    project2_prediction_uni_admission()
elif st.session_state.selected_project == "Project 3":
    st.header("Project 3")
    st.write("Details for Project 3 will go here.")
else:
    st.write("Please select a project from the sidebar to get started.")
