import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
import streamlit as st
from analysis import generate_code, suggest_improvements

from dotenv import load_dotenv
load_dotenv()
st.set_page_config("ML Model", layout="wide")
st.title("Machine Learning Model")
st.subheader("Upload your dataset 📑")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file:
    st.markdown('### Preview')
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())
    target = st.selectbox(":blue[Select the target variable]", df.columns)
    st.write(f':red(Target Variable :] {target})')
    
    if target:
        
        x=df.drop(columns=[target]).copy()
        y=df[target].copy()
        
        num=x.select_dtypes(include=np.number).columns.tolist()
        cat=x.select_dtypes(include=['object']).columns.tolist()
        
        x[num]=x[num].fillna(x[num].median())
        x[cat]=x[cat].fillna('Missing data')
        
        x=pd.get_dummies(data=x, columns=cat, drop_first=True, dtype=int)
        if y.dtype=='object':
            le=LabelEncoder()
            y=le.fit_transform(y)
            
        if df[target].dtype=='object' or len(np.unique(df[target]))<=10:
            problen_type='classification'
        else:
            problen_type='Regression'
        
        st.write(f':red(Problem Type :] {problen_type})')
        
        xtrain, xtest, ytrain, ytest = train_test_split(x, y,test_size=0.2,random_state=42)
        
        for i in xtrain.columns:
            s=StandardScaler()
            xtrain[i]=s.fit_transform(xtrain[[i]])
            xtest[i]=s.transform(xtest[[i]])
            
        results=[]
            
        if problen_type=='Regression':
            models={'Linear Regression': LinearRegression(),
                   'Random Forest Regressor': RandomForestRegressor(random_state=42),
                   'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42)}
            
            for name, model, in models.items():
                model.fit(xtrain, ytrain)
                ypred=model.predict(xtest)
                
                results.append({'Model name': name,
                                'MSE': round(mean_squared_error(ytest, ypred),3),
                                'R2 Score': round(r2_score(ytest, ypred),3),
                                'RMSE': round(np.sqrt(mean_squared_error(ytest, ypred)),3)})
        else:
            models={'Logistic Regression': LogisticRegression(random_state=42),
                   'Random Forest Classifier': RandomForestClassifier(random_state=42),
                   'Gradient Boosting Classifier': GradientBoostingClassifier(random_state=42)}
            
            for name, model, in models.items():
                model.fit(xtrain, ytrain)
                ypred=model.predict(xtest)
                
                results.append({'Model name': name,
                                'Accuracy': round(accuracy_score(ytest, ypred),3),
                                'Precision': round(precision_score(ytest, ypred, average='weighted'),3),
                                'Recall': round(recall_score(ytest, ypred, average='weighted'),3),
                                'F1 Score': round(f1_score(ytest, ypred, average='weighted'),3)})
        results_df=pd.DataFrame(results)
        st.write('### :green[Results]')
        st.dataframe(results_df)
        
        if problen_type=='Regression':
            st.bar_chart(results_df.set_index('Model name')[['MSE', 'R2 Score', 'RMSE']])
            st.bar_chart(results_df.set_index('Model name')[['F1 Score', 'Precision', 'Recall', 'Accuracy']])
        else:
            st.bar_chart(results_df.set_index('Model name')['Accuracy'])
            st.bar_chart(results_df.set_index('Model name')['F1 Score'])
                
     # AI insights
     
        if st.button('Generate Summary'):
           summary = generate_code(results_df)
           st.write(summary)
           
        if st.button('Suggest Improvements'):
              improvements = suggest_improvements(results_df)
              st.write(improvements)
            