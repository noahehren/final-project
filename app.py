#from operator import index
#from re import S
#from turtle import color
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgbm

import pickle
from google.cloud import firestore

import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "final-project-348821-39ba920f1e3e.json"
db = firestore.Client()

st.set_page_config(page_title="My Streamlit App", layout="wide")
st.markdown('''
<style>
    #MainMenu
    {
        display: none;
    }
    .css-18e3th9, .css-1d391kg
    {
        padding: 1rem 2rem 2rem 2rem;
    }
</style>
''', unsafe_allow_html=True)

st.header("Predicting Diabetes based on Risk Factors")
st.sidebar.header("MLExperts")
st.sidebar.markdown('*\'Analytics done right!\'*')
st.sidebar.image('diabetes.png')
page_selected = st.sidebar.radio("Menu", ["Home", "Model", "About"])

if page_selected == 'Home':

    docs = db.collection(u'diabetes').stream()
    items = []
    for doc in docs:
        items.append(doc.to_dict())
    df = pd.DataFrame.from_records(items)

    st.markdown('Our analytics application uses machine learning models to predict the presence of diabetes in patients surveyed from the CDC\'s Behavioral Risk Factor Surveillance System (BRFSS).')
    st.subheader('Our Big Story Plot')
    col1, col2 = st.columns((2,1))
    with col1:
        fig, ax = plt.subplots(figsize = (6,3))
        labels = np.array([1,0])
        ax.bar(labels, 
                df["Diabetes_Predictions"].astype(int).value_counts().sort_values(ascending=True), 
                width=0.6, 
                color = ('#666666', '#00008b'))
        ax.set_xticks(labels)
        ax.set_xticklabels(['No Diabetes', 'Diabetes/Prediabetes'])
        ax.set_xlabel('Prediction')
        ax.set_ylabel('Count of Occurrences')
        st.pyplot(fig)
    with col2:
        st.markdown('This plot shows the total count of patients predicted to have diabetes/prediabetes and to not have diabetes.')

    st.subheader('Count of Diabetes based on Predictor Variable(s)')

    col1, col2 = st.columns((2,1))
    st.sidebar.subheader('Select Predictor Variable')
    with col1:
        Variable = st.sidebar.multiselect('Predictor Variable(s)', df.drop(columns = ['Diabetes_Predictions', 'BMI', 'PhysHlth']).columns.sort_values(), default = 'Age')
        print(Variable)
        #df.loc[df[str(Variable)]]['Diabetes_Predictions'].value_counts() #ask yatish how we can index based on multiple variables
        for var in Variable:      
            fig, ax = plt.subplots(figsize = (6,2))      
            ax = pd.crosstab(df[var].astype(int), df['Diabetes_Predictions'].astype(int)).plot(
                    kind="bar", 
                    figsize=(6,2), 
                    xlabel = var,
                    ylabel = 'Count of Occurrences',
                    rot = 0,
                    color = ('#00008b', '#666666')
                    )
            legend = plt.legend()
            legend.get_texts()[0].set_text('No Diabetes')
            legend.get_texts()[1].set_text('Diabetes/Prediabetes')
            st.pyplot(ax.figure)
    with col2:
        st.markdown('This plot(s) shows the total count of patients predicted to have diabetes/prediabetes and no diabetes with resepect to selected independent (predictor) variables. We removed the BMI and PhysHlth variables because they contain too many unique data points.')
        st.markdown('In most cases, x-axis label \'0\' inidicates that a person does not do/have something (for example, \'0\' for HighBP means that person does not have a high blood pressure); and \'1\' indicates that a person has does do/have something. For the \'Gender\' variable, \'0\' inidicates Female and \'1\' indicates Male.')
    
    st.subheader('Model Performance')
    df_sample = df.sample(5)
    for index, row in df_sample.iterrows():
        col1, col2 = st.columns((2,3))
        with col1:
            if row['Diabetes_Predictions'] == 1:
                st.success('Does not have diabetes/prediabetes')
            else:
                st.error('Does have diabetes/prediabetes')
        with col2:
            rdf = pd.DataFrame([{
                'Age': row['Age'],
                'Sex': row['Sex'],
                'Education': row['Education'],
                'Income':row['Income'],
                'BMI': row['BMI'],
                'HighBP':row['HighBP'],
                'HighChol':row['HighChol']
            }])
            st.dataframe(rdf)

elif page_selected == 'Model':
    df = pd.read_csv('diabetes_train.csv')
    st.subheader('Background of Our Dataset')
    st.markdown('Our dataset contains survey data from the CDC\'s Behavioral Risk Factor Surveillance System (BRFSS), a health-related telephone survey which collects health data from thousands of Americans.')
    st.markdown('The survey asked surveyee\'s questions regarding 21 potential risk factors for diabetes, such as BMI, cholesterol levels, smoking frequency, and physical activity. It also asked surveyee\'s if they have ever had diabetes.')
    st.markdown('Based on this data, our model aims to predict whether a person has diabetes based on the 21 risk factors in teh survey.')
    st.write(df.head(2))
    st.caption('*0 = low/no*' '$\;\;\;\;\;\;$' '*1 = high/yes*')
    st.caption('*Variables with multiple categorical values (Age, Education, etc.) have unique classification values.*')

    st.subheader('Model Building')
    st.markdown('**Feature Selection**')
    st.markdown('We used all 21 input variables from the original dataset to predict diabetes with our model. All of the original input variables are relevant to predicting our target variable.')
    st.markdown('**Model Selection**')
    st.markdown('We tested five different models to find the optimal model to predict our target variable: Decision Tree Classifier, KNeighbors Classifier, Random Forest Classifier, XGBoost, and Light GBM.')
    st.markdown('To determine the best model, we ran GridSearchCV on all the classifiers to find the best parameters for each.') 
    st.markdown('Then, we calculated the cross-validation accuracy, testing accuracy, and confusion matrix for each classifier. LightGBM consistently produced the highest cross-validation and testing accuracy out of all the classifiers (performing only marginally better than XGBoost). Furthermore, the margin between the cross-validation accuracy and test accuracy for LightGBM was miniscule, which gave us confidence that the accuracy score was reliable.') 
    st.markdown('To ensure that LightGBM was the best model, we plotted the ROC curves for all the classifiers. We found that LightGBM was still the best model because it has the highest AUC score.')

    st.subheader('Test a Classifier')
    st.markdown('To validate our aforementioned statements, we included a feature where you can see all five of our model performances.')
    st.markdown('Choose a classifier and the plot(s) you want to see. After making your selection, press \'Run Model\'. The results will be printed below.')
    st.markdown('Please note that a \'0\' represents **no diabetes** and a \'1\' represents **prediabetes/diabetes**.')

    #@st.cache(persist = True)
    @st.experiment_memo
    def split(df2):
        X = df2.drop(columns = ['Diabetes_binary'])
        X = pd.get_dummies(X, columns=['BMI', 'GenHlth', 'PhysHlth', 'Age', 'Education', 'Income'],drop_first=True)
        y = df2['Diabetes_binary']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        return X_train, X_test, y_train, y_test
    
    def plot_metrics(metrics_list):
        st.set_option('deprecation.showPyplotGlobalUse', False)

        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix") 
            plot_confusion_matrix(model, X_test, y_test)
            st.pyplot()
        
        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model, X_test, y_test)
            st.pyplot()

    df2 = pd.read_csv('diabetes_train.csv')
    X_train, X_test, y_train, y_test = split(df2)

    st.sidebar.subheader("Test a Classifier")
    classifier = st.sidebar.selectbox("What classifier do you want to test?", ("Decision Tree", "KNearestNeighbor", "LightGBM", "Random Forest", "XGBoost"))

    if classifier == 'Decision Tree':
        metrics = st.sidebar.multiselect("What metrics to plot?",('Confusion Matrix', 'ROC Curve'))

        if st.sidebar.button("Run Model", key='model'):
            st.subheader("Decision Tree (Test Results)")
            with st.spinner("Drawing your plots!"):
                model = DecisionTreeClassifier(criterion = 'gini', splitter = 'best', max_depth = 3, max_leaf_nodes= 8)
                model.fit(X_train, y_train)
                modelscore = model.fit(X_train, y_train)
                accuracy = model.score(X_test, y_test)
                scores = cross_val_score(modelscore, X_train, y_train, scoring= 'accuracy', cv = 5)
                y_pred = model.predict(X_test)
                st.write("Training Accuracy ", accuracy.round(2))
                st.write("CrossVal Accuracy", np.mean(scores.round(2)))
                plot_metrics(metrics)

    if classifier == 'KNearestNeighbor':
        metrics = st.sidebar.multiselect("What metrics to plot?",('Confusion Matrix', 'ROC Curve'))

        if st.sidebar.button("Run Model", key='model'):
            st.subheader("KNearestNeighbor (Test Results)")
            with st.spinner("Drawing your plots!"):
                model = KNeighborsClassifier(n_neighbors = 9, weights = 'uniform')
                model.fit(X_train, y_train)
                modelscore = model.fit(X_train, y_train)
                accuracy = model.score(X_test, y_test)
                scores = cross_val_score(modelscore, X_train, y_train, scoring= 'accuracy', cv = 5)
                y_pred = model.predict(X_test)
                st.write("Training Accuracy ", accuracy.round(2))
                st.write("CrossVal Accuracy", np.mean(scores.round(2)))
                plot_metrics(metrics)

    if classifier == 'Random Forest':
        metrics = st.sidebar.multiselect("What metrics to plot?",('Confusion Matrix', 'ROC Curve'))

        if st.sidebar.button("Run Model", key='model'):
            st.subheader("Random Forest (Test Results)")
            with st.spinner("Drawing your plots!"):
                model = RandomForestClassifier(criterion = 'entropy', n_estimators = 140)
                model.fit(X_train, y_train)
                modelscore = model.fit(X_train, y_train)
                accuracy = model.score(X_test, y_test)
                scores = cross_val_score(modelscore, X_train, y_train, scoring= 'accuracy', cv = 5)
                y_pred = model.predict(X_test)
                st.write("Training Accuracy ", accuracy.round(2))
                st.write("CrossVal Accuracy", np.mean(scores.round(2)))
                plot_metrics(metrics)

    if classifier == 'XGBoost':
        metrics = st.sidebar.multiselect("What metrics to plot?",('Confusion Matrix', 'ROC Curve'))

        if st.sidebar.button("Run Model", key='model'):
            st.subheader("XGBoost (Test Results)")
            with st.spinner("Drawing your plots!"):
                model = xgb.XGBClassifier(max_depth = 4, n_estimators = 100, labael_encoder = False)
                model.fit(X_train, y_train)
                modelscore = model.fit(X_train, y_train)
                accuracy = model.score(X_test, y_test)
                scores = cross_val_score(modelscore, X_train, y_train, scoring= 'accuracy', cv = 5)
                y_pred = model.predict(X_test)
                st.write("Training Accuracy ", accuracy.round(2))
                st.write("CrossVal Accuracy", np.mean(scores.round(2)))
                plot_metrics(metrics)

    if classifier == 'LightGBM':
        metrics = st.sidebar.multiselect("What metric(s) to plot?",('Confusion Matrix', 'ROC Curve'))

        if st.sidebar.button("Run Model", key='model'):
            st.subheader("LightGBM (Test Results)")
            with st.spinner("Drawing your plots!"):
                model = lgbm.LGBMClassifier(max_depth = 9, num_leaves = 21)
                model.fit(X_train, y_train)
                modelscore = model.fit(X_train, y_train)
                accuracy = model.score(X_test, y_test)
                scores = cross_val_score(modelscore, X_train, y_train, scoring= 'accuracy', cv = 5)
                y_pred = model.predict(X_test)
                st.write("Training Accuracy ", accuracy.round(2))
                st.write("CrossVal Accuracy", np.mean(scores.round(2)))
                plot_metrics(metrics)

###################################### NEED TO FIX THIS CODE ####################################
    st.subheader('Data Exploration Graphs')
    st.markdown('Our dataset was cleaned prior to our exploration/analysis and its columns only contained categorical data. Therefore, our exploration consisted of observing the value counts of each variable.')
    st.markdown('*The \'Diabetes_binary\', our target variable, selection is balanced, which highlights how clean the dataset already was.*')
    st.sidebar.subheader("Choose Variable for Histogram")
    
    Variable = st.sidebar.selectbox('Variable', df2.columns.sort_values(), index = 4)
    fig, ax = plt.subplots(figsize = (8,2))
    ax.bar(
        x = np.unique(df[Variable].values.astype(int)), 
        height = df[Variable].astype(int).value_counts(), 
        align = 'center', 
        tick_label = np.unique(df[Variable].values.astype(int)),
        width = 0.3,
        color = ('#00008b', '#666666')
        )
    ax.set_ylabel('Number of Occurences', fontsize = 7)
    ax.set_xlabel(Variable, fontsize = 7)
    plt.xticks(fontsize = 7) 
    plt.yticks(fontsize = 7)
    #ax.bar_label(df[Variable].astype(int).value_counts()) ask yatish about data labels
    st.pyplot(fig)#.figure)
    
#################################################################################################

    st.subheader('Test our Model with Your Data!')
    st.markdown('In this section, you can test our prediction model with your own data!')
    st.markdown(' Simply select the values you want from the boxes below (additional variables in the \'Other Predictor Variables\' expander) and then press the **prediction button** in the sidebar.')
    col1, col2 = st.columns(2)
    with col1:
        Sex_options = { 
        0: "Female",
        1: "Male",
        }       

        Sex = st.selectbox(
            label="Select your Sex:",
            options= (0, 1), 
            format_func=lambda x: Sex_options.get(x),
            index = 0
        )
    
        Age_options = { 
        1: "I am between the ages of 18-24",
        2: "I am between the ages of 25-29",
        3: "I am between the ages of 30-34",
        4: "I am between the ages of 35-39",
        5: "I am between the ages of 40-44",
        6: "I am between the ages of 45-49",
        7: "I am between the ages of 50-54",
        8: "I am between the ages of 55-59",
        9: "I am between the ages of 60-64",
        10: "I am between the ages of 65-69",
        11: "I am between the ages of 70-74",
        12: "I am between the ages of 75-79",
        13: "I am 80 years old or older"
        }   

        Age = st.selectbox(
            label="Select your Age Group:",
            options= (1,2,3,4,5,6,7,8,9,10,11,12,13), 
            format_func=lambda x: Age_options.get(x),
            index = 1
            )
    
        BMI = st.number_input(
            label="Please input your BMI (integer only):",
            min_value = 5,
            max_value= 200, 
            value = 20
            )  

    with col2: 
        HighBP_options = { 
        0: "I do not have high blood pressure",
        1: "I do have high blood pressure"
        }   

        HighBP = st.selectbox(
            label="Select a Blood Pressure option:",
            options= (0, 1), 
            format_func=lambda x: HighBP_options.get(x),
            index = 0
            )   
        
        HighChol_options = { 
        0: "I do not have high cholesterol",
        1: "I do have high cholesterol",
        }   

        HighChol = st.selectbox(
            label="Select a Cholesterol option:",
            options= (0, 1), 
            format_func=lambda x: HighChol_options.get(x),
            index = 0
            )

        PhysActivity_options = { 
        0: "I have not been physically activity in past 30 days (not including job)",
        1: "I have been physically activity in past 30 days (not including job)",
        }   

        PhysActivity = st.selectbox(
            label="Select a Physical Activity option:",
            options= (0, 1), 
            format_func=lambda x: PhysActivity_options.get(x),
            index = 1
            )       

    with st.expander("Other Predictor Variables"):
        col1, col2, col3 = st.columns(3)

        HeartDiseaseorAttack_options = { 
        0: "I do not have coronary heart disease (CHD) or myocardial infarction (MI)",
        1: "I do have coronary heart disease (CHD) or myocardial infarction (MI)",
        }   

        HeartDiseaseorAttack = st.selectbox(
            label="Select a Heart Disease/Attack option:",
            options= (0, 1), 
            format_func=lambda x: HeartDiseaseorAttack_options.get(x),
            index = 0
            )    
        
        NoDocbcCost_options = { 
        0: "In the past 12 months, when I needed to see a doctor, I was \'not prevented\' from going because of the cost",
        1: "In the past 12 months, when I needed to see a doctor, I was \'prevented\' from going because of the cost",
        }   

        NoDocbcCost = st.selectbox(
            label="Select a Doctor Cost option:",
            options= (0, 1), 
            format_func=lambda x: NoDocbcCost_options.get(x),
            index = 0
            ) 

        AnyHealthcare_options = { 
        0: "I do not have any kind of health care coverage, including health insurance, prepaid plans such as HMO, etc.",
        1: "I do have health care coverage, including health insurance, prepaid plans such as HMO, etc.",
        }   

        AnyHealthcare = st.selectbox(
            label="Select a Healthcare option:",
            options= (0, 1), 
            format_func=lambda x: AnyHealthcare_options.get(x),
            index = 1
            ) 

        Education_options = { 
        1: "I never attended school or only kindergarten",
        2: "I completed grades 1-8",
        3: "I completed grades 9-11 (some high school)",
        4: "I completed grade 12 or GED (high school graduate)",
        5: "I completed 1 to 3 years of college (some college of technical school",
        6: "I completed 4 years or more of college (college graduate)",
        }   

        Education = st.selectbox(
            label="Select your Education level:",
            options= (1,2,3,4,5,6), 
            format_func=lambda x: Education_options.get(x),
            index = 4
            )
        
        Income_options = { 
        1: "I make less than or equal to $10,000 per year",
        2: "I make more than $10,000 but less than or equal to $15,000 per year",
        3: "I make more than $15,000 but less than or equal to $20,000 per year",
        4: "I make more than $20,000 but less than or equal to $25,000 per year",
        5: "I make more than $25,000 but less than or equal to $30,000 per year",
        6: "I make more than $35,000 but less than or equal to $50,000 per year",
        7: "I make more than $50,000 but less than or equal to $75,000 per year",
        8: "I make more than $75,000 per year",
        }   

        Income = st.selectbox(
            label="Select your Income:",
            options= (1,2,3,4,5,6,7,8), 
            format_func=lambda x: Income_options.get(x),
            index = 6
            )

        HvyAlcoholConsump_options = { 
        0: "I am not a heavy alcohol drinker (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week)",
        1: "I am a heavy alcohol drinker (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week)",
        }   

        HvyAlcoholConsump = st.selectbox(
            label="Select a Alcohol Consumption option:",
            options= (0, 1), 
            format_func=lambda x: HvyAlcoholConsump_options.get(x),
            index = 0
            ) 

        with col1:
            CholCheck_options = { 
            0: "I have not gotten my cholesterol checked in the past 5 years",
            1: "I have gotten my cholesterol checked in the past 5 years",
            }   

            CholCheck = st.selectbox(
                label="Select a Cholesterol Check option:",
                options= (0, 1), 
                format_func=lambda x: CholCheck_options.get(x),
                index = 1
                )   

            Smoker_options = { 
            0: "I have not smoked at least 100 cigarettes (5 packs) in my life",
            1: "I have smoked at least 100 cigarettes (5 packs) in my life",
            }   

            Smoker = st.selectbox(
                label="Select a Smoking option:",
                options= (0, 1), 
                format_func=lambda x: Smoker_options.get(x),
                index = 0
                )  

            Stroke_options = { 
            0: "I have not had a stroke",
            1: "I have had a stroke",
            }   

            Stroke = st.selectbox(
                label="Select a Stroke option:",
                options= (0, 1), 
                format_func=lambda x: Stroke_options.get(x),
                index = 0
                )   

 
        with col2:
            Veggies_options = { 
            0: "I do not consume vegetables one or more times per day",
            1: "I do consume vegetables one or more times per day",
            }   

            Veggies = st.selectbox(
                label="Select a Vegetables option:",
                options= (0, 1), 
                format_func=lambda x: Veggies_options.get(x),
                index = 1
                )   

            GenHlth_options = { 
            1: "I have excellent general health",
            2: "I have very good general health",
            3: "I have good general health",
            4: "I have fair general health",
            5: "I have poor general health",
            }   

            GenHlth = st.selectbox(
                label="Select a General Health option:",
                options= (1,2,3,4,5), 
                format_func=lambda x: GenHlth_options.get(x),
                index = 0
                )

            MentHealth = st.number_input(
                label="For how many days during the past 30 days was your mental health (includes stress, depression, and problems with emotions) not good?:",
                min_value = 0,
                max_value= 30, 
                value = 7
                )
        
        with col3:
            Fruits_options = { 
            0: "I do not consume fruit one or more times per day",
            1: "I do consume fruit one or more times per day",
            }   

            Fruits = st.selectbox(
                label="Select a Fruits option:",
                options= (0, 1), 
                format_func=lambda x: Fruits_options.get(x),
                index = 1
                )   

            DiffWalk_options = { 
            0: "I do not have serious difficulty walking or climbing stairs",
            1: "I do have serious difficulty walking or climbing stairs",
            }   

            DiffWalk = st.selectbox(
                label="Select a Difficulty Walking option:",
                options= (0, 1), 
                format_func=lambda x: DiffWalk_options.get(x),
                index = 0
                ) 

            PhysHealth = st.number_input(
                label="For how many days during the past 30 days was your physical health (includes physical illness and injury) not good?:",
                min_value = 0,
                max_value= 30, 
                value = 0
                ) 

    testdata = pd.DataFrame({
        'Diabetes_binary':0,
        'HighBP':HighBP,
        'HighChol':HighChol,
        'CholCheck':CholCheck,
        'BMI':BMI,
        'Smoker':Smoker,
        'Stroke':Stroke,
        'HeartDiseaseorAttack':HeartDiseaseorAttack,
        'PhysActivity':PhysActivity,
        'Fruits':Fruits,
        'Veggies':Veggies,
        'HvyAlcoholConsump':HvyAlcoholConsump,
        'AnyHealthcare':AnyHealthcare,
        'NoDocbcCost':NoDocbcCost,
        'GenHlth':GenHlth,
        'MentHlth':MentHealth,
        'PhysHlth':PhysHealth,
        'DiffWalk':DiffWalk,
        'Sex':Sex,
        'Age':Age,
        'Education':Education,
        'Income':Income
        }, index = [0])

    pipeline_file = "pipe_lightgbm.pkl"
    def predict_diabetes(data):
        df = data
        pipeline = pickle.load(open(pipeline_file,'rb'))
        predicted = pipeline.predict(df)
        df['Prediction'] = predicted
        return df
    
    st.sidebar.subheader("Your Diabetes Prediction")
    if st.sidebar.button("Prediction", key='prediction'):
        prediction = predict_diabetes(testdata)
        if prediction['Prediction'].any() == 1:
            st.sidebar.markdown('Our model predicts that you **have diabetes or are prediabetic**')
        else:
            st.sidebar.markdown('Our model predicts that you **do not have diabetes**')
            st.balloons()

else:
    st.subheader('About Us')
    col1, col2 = st.columns((1,40))
    with col2:
        st.markdown("Our company, MLExperts, is dedicated to creating machine learning models. Our team of highly skilled individuals works in over 40 countries to help clients deliver on their prediction needs. Please contact our team at info@mlexperts.com to learn more about our services.")

    col1, col2 = st.columns((1,40))
    with col2:
        st.markdown('**Noah Ehren**')
    col11, col12, col13 = st.columns((1,6,34))
    with col12:
        st.image('noah.jpg') 
    with col13:
        st.write('Noah graduated from Tulane University with a Master of Science in Business Analytics in 2017. Since then, he has worked on analytics teams at major companies around the globe. In his free time, Noah enjoys playing tennis.')

    col1, col2 = st.columns((1,40))
    with col2:
        st.markdown('**Kristen O\'Handley**')
    col11, col12, col13 = st.columns((1,6,34))
    with col12:
        st.image('kristen1.jpg') 
    with col13:
        st.write('Kristen graduated with her Master of Science in Business Analytics from Tulane University in 2020. Since then she has worked in top hospitals across the country as an analyst to combat the diabetes pandemic in America. Kristen enjoys sky diving and playing extreme sports in her free time')

    col1, col2 = st.columns((1,40))
    with col2:
        st.markdown('**Kristen Tobin**')
    col11, col12, col13 = st.columns((1,6,34))
    with col12:
        st.image('kristen2.jpg') 
    with col13:
        st.write('After graduating from Tulane University with a Master of Business Analytics degree in 2018, Kristen began her career in marketing analytics in London. Kristen has enjoyed the opportunity to work in England and the United States over the years. Currently residing in the US, Kristen enjoys spending time with her family.')

    col1, col2 = st.columns((1,40))
    with col2:
        st.markdown('**Josh Quigley**')
    col11, col12, col13 = st.columns((1,6,34))
    with col12:
        st.image('josh.jpg')
    with col13:
        st.write('Josh graduated from Wake Forest University with a Master of Science in Business Analytics in 2017. Since then, he has been an analytics consultant for the CIA. In his free time, Josh enjoys racing cars.')
