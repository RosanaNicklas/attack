#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importar librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# In[2]:


# cargar datos
df = pd.read_csv('heart.csv')


# In[3]:


# previsualizar dataframe
df.head(5)


# In[4]:


# Dimensiones y primeras filas
df.shape


# In[5]:


# Tipos y valores faltantes
df.info()


# In[6]:


# sacar columnas
df.columns


# In[7]:


# numero defilas
df.index


# In[8]:


# tipos de datos por columnas
df.dtypes


# In[9]:


# Estadísticos descriptivos
df.describe(include='all')


# In[10]:


print("Number of Unique Values for Each Feature:\n")
print(df.nunique())


# About Dataset
# 
# Age : Age of the patient
# 
# Sex: The person’s sex (1 = male, 0 = female)
# 
# cp: chest pain type
# 
# — Value 0: asymptomatic
# 
# — Value 1: atypical angina
# 
# — Value 2: non-anginal pain
# 
# — Value 3: typical angina
# 
# trestbps: The person’s resting blood pressure (mm Hg on admission to the hospital)
# 
# chol: The person’s cholesterol measurement in mg/dl
# 
# fbs: The person’s fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)
# 
# restecg: resting electrocardiographic results
# 
# — Value 0: showing probable or definite left ventricular hypertrophy by Estes’ criteria
# 
# — Value 1: normal
# 
# — Value 2: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
# 
# thalach: The person’s maximum heart rate achieved
# 
# exang: Exercise induced angina (1 = yes; 0 = no)
# 
# oldpeak: ST depression induced by exercise relative to rest (‘ST’ relates to positions on the ECG plot. See more here)
# 
# slope: the slope of the peak exercise ST segment — 0: downsloping; 1: flat; 2: upsloping
# 
# target: Heart disease (1 = no, 0= yes)
# 
# 

# In[11]:


# Print value counts for each column
for col in df.columns:
    print('---🔢',df[col].value_counts())
    print('-'*40)


# In[12]:


# Distribución de la variable objetivo
df['HeartDisease'].value_counts()


# In[13]:


# valores nan
df.isnull().sum()


# In[14]:


rows_with_missing = df[df.isna().any(axis=1)]

from IPython.display import display
display(rows_with_missing)


# In[15]:


# valores duplicados
df.duplicated().sum()


# In[16]:


# Convertir todas las columnas categóricas a numéricas
# Versión segura para todas las columnas
df['Sex'] = df['Sex'].map({'M': 0, 'F': 1, 'M': 0, 'F': 1}).fillna(-1).astype('int64')
df['ChestPainType'] = df['ChestPainType'].map({
    'TA': 0, 
    'ATA': 1,
    'NAP': 2,
    'ASY': 3
}).fillna(-1).astype('int64')


df['ST_Slope'] = df['ST_Slope'].map({
    'Up': 0,
    'Flat': 1,
    'Down': 2
}).astype('int64')

df['ExerciseAngina'] = df['ExerciseAngina'].map({'N': 0, 'Y': 1}).astype('int64')

df['RestingECG'] = df['RestingECG'].map({
    'Normal': 0,
    'ST': 1,
    'LVH': 2
}).astype('int64')


# In[17]:


df.rename(columns={'cp':'chest_pain_type',
                    'trestbps':'resting_bp',
                    'chol':'cholesterol',
                    'FastingBS':'fasting_blood_sugar',
                    'RestingECG':'rest_ecg_result',
                    'MaxHR':'max_heart_rate',
                    'ExerciseAngina':'exercise_induced_angina',
                    'Oldpeak':'st_depression',
                    'ST_Slope':'st_slop',
                    'ca':'num_major_vessels',
                    'HeartDisease':'heart_disease'},inplace=True)


# In[18]:


df.columns


# In[19]:


df.dtypes


# In[20]:


# Convert 'Sex' to numeric values
df['Sex_numeric'] = df['Sex'].map({'male': 0, 'female': 1})

# Calculate skewness and kurtosis
skewness = df['Sex_numeric'].skew()
kurtosis = df['Sex_numeric'].kurtosis()  # This calculates excess kurtosis


# In[21]:


# Print the results
print(f"Skewness of Sex: {skewness:.2f}")
print(f"Sex → skewness: {skewness:.2f}, excess kurtosis: {kurtosis:.2f}")



# In[22]:


# Seleccionar solo columnas numéricas
numeric_df = df.select_dtypes(include=['number'])

# Calcular y ordenar skewness
skews = numeric_df.skew().sort_values(ascending=False)
print("Skewness por variable (orden descendente):\n", skews)

# Filtrar variables con alta asimetría
high_skew = skews[abs(skews) > 1]
if not high_skew.empty:
    print("\n¡Alerta! Variables con asimetría significativa (|skew| > 1):\n", high_skew)
else:
    print("\nNo hay variables con asimetría significativa (|skew| <= 1).")

# Opcional: Calcular kurtosis
kurtosis = numeric_df.kurtosis().sort_values(ascending=False)
print("\nExcess kurtosis por variable:\n", kurtosis)


# In[23]:


df.drop(labels='Sex_numeric', axis=1, inplace=True)


# In[24]:


import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
#continous numerical variables

# Select only continuous/numeric features for scatter matrix
numeric_features = ['Age', 'RestingBP', 'Cholesterol','st_depression']
#categorical numerical variables
cat=['Sex', 'ChestPainType','fasting_blood_sugar','rest_ecg_result','exercise_induced_angina','st_slop', 'max_heart_rate']
# Create scatter matrix
scatter_matrix(df[numeric_features], figsize=(12, 12), diagonal='hist', alpha=0.8)
plt.suptitle('Pairwise Scatter Matrix of Continuous Variables', y=1.02)
plt.tight_layout()
plt.show()


# In[25]:


import seaborn as sns

category=[('ChestPainType', ['typical','nontypical','nonanginal','asymptomatic'])]
continuous = [('Age', 'Age in year'),
              ('Sex','1 for Male 0 for Female'),
              ('RestingBP','BP in Rest State'),
              ('fasting_blood_sugar','Fasting blood glucose'),
              ('rest_ecg_result','ECG at rest'),
              ('Cholesterol', 'serum cholestoral in mg/d'),
              ('max_heart_rate','Max Heart Rate'),
              ('exercise_induced_angina','Exchange Rate'),
              ('st_slope','Slope of Curve'),
              ('st_detresion', 'ST depression by exercise relative to rest')]



def plotCategorial(attribute, labels, ax_index):
    sns.countplot(x=attribute, data=df, ax=axes[ax_index][0])
    sns.countplot(x='heart_disease', hue=attribute, data=df, ax=axes[ax_index][1])
    avg = df[[attribute, 'heart_disease']].groupby([attribute], as_index=False).mean()
    sns.barplot(x=attribute, y='heart_disease', hue=attribute, data=avg, ax=axes[ax_index][2])

    for t, l in zip(axes[ax_index][1].get_legend().texts, labels):
        t.set_text(l)
    for t, l in zip(axes[ax_index][2].get_legend().texts, labels):
        t.set_text(l)


def plotContinuous(attribute, xlabel, ax_index):
    sns.distplot(df[[attribute]], ax=axes[ax_index][0])
    axes[ax_index][0].set(xlabel=xlabel, ylabel='density')
    sns.violinplot(x='heart_disease', y=attribute, data=df, ax=axes[ax_index][1])


def plotGrid(isCategorial):
    if isCategorial:
        [plotCategorial(x[0], x[1], i) for i, x in enumerate(category)] 
    else:
        [plotContinuous(x[0], x[1], i) for i, x in enumerate(continuous)]


# In[26]:


#fig_categorial,axes=plt.subplots(nrows=len(category), ncols=3, figsize=(10, 10))
#plotGrid(isCategorial=True)


# In[27]:


#fig_continuous, axes = plt.subplots(nrows=len(continuous), ncols=2, figsize=(10,10))
#plotGrid(isCategorial=False)


# In[28]:


# matriz correlaciones
# calculate the correlation matrix on the numeric columns
corr = df.corr()
sns.heatmap(corr, annot=True)


# Dataset contains more patient records of age around 55-65
# Male patient records are more compared to female ones
# Chest pain type of asymptomatic has been common from patient records
# The person’s resting blood pressure values are common between 120-140 mm Hg
# Common person’s cholesterol measurement in mg/dl is between 200-300 with highest around 250
# The person’s fasting blood sugar < 120 mg/dl is more
# Slope of peak exercise has more records with downsloping value
# The number of major vessels recorded here is 0
# Most people had normal blood flow and few had reversible defect (a blood flow is observed but it is not normal) related to thalassemia record
# Most of the resting electrocardiographic results shows probable or definite left ventricular hypertrophy by Estes’ criteria or normal
# The person’s maximum heart rate achieved are around 140-180 with maximum values around 150-160
# Exercise induced angina in records shows mostly no

# In[29]:


# Cálculo de outliers según el método IQR
Q1 = df['Age'].quantile(0.25)
Q3 = df['Age'].quantile(0.75)
IQR = Q3 - Q1


# In[30]:


lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR


# In[31]:


outliers = df[(df['Age'] < lower_bound) | (df['Age'] > upper_bound)]['Age']


# In[32]:


print(f"Número de outliers detectados: {outliers.shape[0]}")
print("Rango IQR:", lower_bound, "–", upper_bound)
print("Ejemplos de valores outliers:", outliers.unique()[:10])


# In[33]:


X_shape =df.iloc[:,0:11]
Y_shape =df['heart_disease']


# In[34]:


import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import OrdinalEncoder

# Assuming:
# X_shape is a DataFrame with categorical features
# Y_shape contains the target variable

# 1. Encode categorical features
encoder = OrdinalEncoder()
X_encoded = encoder.fit_transform(X_shape)

# Convert back to DataFrame to maintain column names
X_encoded = pd.DataFrame(X_encoded, columns=X_shape.columns)

# 2. Initialize and fit the model
model = ExtraTreesClassifier(
    n_estimators=100,
    random_state=42,  # for reproducibility
    n_jobs=-1  # use all available cores
)

model.fit(X_encoded, Y_shape)

# 3. Get feature importances
feature_importances = model.feature_importances_

# 4. Create a DataFrame for better visualization
importance_df = pd.DataFrame({
    'Feature': X_shape.columns,
    'Importance': feature_importances
}).sort_values('Importance', ascending=False)

print("Feature Importances:")
print(importance_df)

# 5. Plotting (optional)
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance Score')
plt.title('Feature Importances from ExtraTreesClassifier')
plt.gca().invert_yaxis()  # Most important at top
plt.show()


# In[35]:


#eliminalos columna que no aporta valor
#df.drop('Unnamed: 0',axis=1, inplace=True)
#df.head()


# In[36]:


def plotAge():
    # Create figure and axes
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 8))

    # Plot 1: KDE Plot with fill instead of shade
    sns.kdeplot(data=df, x="Age", hue="heart_disease", fill=True, ax=axes[0])
    axes[0].set(xlabel='Age', ylabel='Density')
    axes[0].legend(title='Heart Disease', labels=['No Disease', 'Disease'])

    # Plot 2: Barplot of disease probability by age
    avg = df.groupby("Age")["heart_disease"].mean().reset_index()
    sns.barplot(x='Age', y='heart_disease', data=avg, ax=axes[1])
    axes[1].set(xlabel='Age', ylabel='Disease Probability')

    plt.tight_layout()
    plt.show()

plotAge()


# In[37]:


plt.figure()
df['ChestPainType'].value_counts().sort_index().plot(kind='bar')
plt.title('Conteo de ChestPainType')
plt.xlabel('Número de ChestPainType')
plt.ylabel('Frecuencia')
plt.show()


# In[38]:


plt.figure()
df['exercise_induced_angina'].value_counts().sort_index().plot(kind='bar')
plt.title('Conteo de exercise_induced_angina')
plt.xlabel('Número de exercise_induced_angina')
plt.ylabel('Frecuencia')
plt.show()


# In[39]:


plt.figure()
df['rest_ecg_result'].value_counts().sort_index().plot(kind='bar')
plt.title('Conteo de rest_ecg_result')
plt.xlabel('Número de rest_ecg_result')
plt.ylabel('Frecuencia')
plt.show()


# In[40]:


plt.figure()
df['st_slop'].value_counts().sort_index().plot(kind='bar')
plt.title('st_slop')
plt.xlabel('st_slop')
plt.ylabel('Frecuencia')
plt.show()


# In[41]:


# gráficamente nº muestras por Real State o Lines en columnnas
sns.countplot(x='fasting_blood_sugar', data=df)
plt.show()


# In[42]:


def plotAge():
    # Create figure and axes
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 8))

    # Plot 1: KDE Plot with fill instead of shade
    sns.kdeplot(data=df, x="Age", hue="heart_disease", fill=True, ax=axes[0])
    axes[0].set(xlabel='Age', ylabel='Density')
    axes[0].legend(title='Heart Disease', labels=['No Disease', 'Disease'])

    # Plot 2: Barplot of disease probability by age
    avg = df.groupby("Age")["heart_disease"].mean().reset_index()
    sns.barplot(x='Age', y='heart_disease', data=avg, ax=axes[1])
    axes[1].set(xlabel='Age', ylabel='Disease Probability')

    plt.tight_layout()
    plt.show()

plotAge()


# In[43]:


def plotAge():
    # Create figure and axes
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 8))

    # Plot 1: KDE Plot with fill instead of shade
    sns.kdeplot(data=df, x="Cholesterol", hue="heart_disease", fill=True, ax=axes[0])
    axes[0].set(xlabel='Cholesterol', ylabel='Density')
    axes[0].legend(title='Heart Disease', labels=['No Disease', 'Disease'])

    # Plot 2: Barplot of disease probability by age
    avg = df.groupby("Cholesterol")["heart_disease"].mean().reset_index()
    sns.barplot(x='Cholesterol', y='heart_disease', data=avg, ax=axes[1])
    axes[1].set(xlabel='Cholesterol', ylabel='Disease Probability')

    plt.tight_layout()
    plt.show()

plotAge()


# In[44]:


# gráficamente nº muestras por clase en pie chart
type_trans = df["Sex"].value_counts()
transactions = type_trans.index
quantity = type_trans.values


# In[45]:


import plotly.express as px

# Make sure 'quantity' and 'transactions' are actual column names in your dataframe
figure = px.pie(df,
             values='Sex',  # Use string column name
             names='heart_disease',  # Use string column name
             hole=0.5,
             title="Ataques al corazon")
figure.show()


# In[46]:


#pip3 install ydata-profiling

# Importamos librerías
import pandas as pd
from ydata_profiling import ProfileReport


# In[47]:


# Generación del informe con ydata-profiling (antes conocido como pandas-profiling)
profile = ProfileReport(
    df,
    title="Informe EDA - HeartAttack",
    explorative=True
)


# In[48]:


# 3. Guardar el informe a HTML
profile.to_file("eda_report.html")


# In[49]:


# 4. Mostrar el informe inline (en Jupyter/Colab)
profile


# In[50]:


#df.drop(labels='Unnamed: 0', axis=1, inplace=True)


# In[51]:


"""print('Pearson Correlation,')
plt.figure(figsize = (11,11))
cor = df.corr().iloc[:,-1:]
sns.heatmap(cor, annot = True, cmap = plt.cm.Blues)
plt.show()

print('abs corr score: ')
print(abs(cor['output'][0:-1]))
cor['output'] = cor['output'][0:-1]
margin = abs(cor['output'][0:-1]).mean()

print('\n')

print('mean {0}'.format(margin))

print('\n')

print('feature selection result: ')
fs = abs(cor['output'][0:-1])[abs(cor['output']) > margin]
print(fs)"""


# In[52]:


import seaborn as sns

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Feature Importances (Ordered)', pad=20)
plt.xlabel('Importance Score', labelpad=15)
plt.ylabel('Feature Name', labelpad=15)
plt.tight_layout()
plt.show()


# In[53]:


# Separa X e y
X = df.drop('heart_disease', axis=1)
y = df['heart_disease']


# In[54]:


#Select top 5 most important features
#selected_features = ['st_slop', 'exercise_induced_angina', 
                 #  'ChestPainType', 'st_depression', 'Cholesterol']
#X_selected = X[selected_features]


# In[55]:


from sklearn.model_selection import train_test_split


# In[56]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[57]:
import pickle
# 2. Save the expected feature names IN ORDER
with open('model_features.pkl', 'wb') as f:
    pickle.dump(list(X_train.columns), f)  # X_train is your training DataFrame

# Forma inicial
print("Antes de limpieza:")
print(" - Shape:", df.shape)
print(" - Índice:", type(df.index), "uniqueness:", df.index.is_unique)


# In[58]:


# recordamos filas duplicadas
df.duplicated().sum()


# In[59]:


# Eliminación de duplicados exactos
df_clean = df.drop_duplicates()
print("Después de drop_duplicates:")
print(" - Shape:", df_clean.shape)
print(" - Filas eliminadas:", df.shape[0] - df_clean.shape[0], "\n")


# In[60]:


# Revisar tipos y ver si hay columnas que convenga convertir
print("Tipos tras limpieza:")
print(df_clean.dtypes)


# In[61]:


# Conteo y porcentaje de nulos
nulos = df_clean.isnull().sum().sort_values(ascending=False)


# In[62]:


porc_nulos = (nulos / len(df_clean) * 100).round(2)


# In[63]:


missing_info = pd.concat([nulos, porc_nulos], axis=1, keys=['n_missing','%_missing'])


# In[64]:


print(missing_info)


# In[65]:


from sklearn.preprocessing import FunctionTransformer, StandardScaler


# In[66]:


print(X_train.describe())


# In[67]:


"""from sklearn.pipeline import Pipeline

# Define preprocessing steps
preprocessor = Pipeline([ 
    ('scaler', StandardScaler())                      # Standardize
])

# Apply to data
X_transformed = preprocessor.fit_transform(X_train)"""


# In[68]:


import seaborn as sns
import matplotlib.pyplot as plt

# Set color palettes
bar_palette = sns.color_palette("pastel")
pie_palette = sns.color_palette("husl", n_colors=10)

# Loop through each column
for column in df.columns:
    # Create figure with 2 subplots (bar & pie)
    fig, axes = plt.subplots(1, 2, figsize=(20, 6), dpi=130)
    fig.suptitle(column, fontsize=16, fontweight='bold', color='#333')

    # --- BAR CHART ---
    sns.countplot(data=df, x=column, hue='Sex', palette='viridis', legend=False)
    # Bar Chart
    sns.countplot(data=df, x=column, ax=axes[0], hue='Sex', palette=bar_palette,width=0.4, legend=False)
    for bar in axes[0].containers:
        axes[0].bar_label(bar, fontsize=10)
    # Add count labels on bars
    #for container in axes[0].containers:
     #   axes[0].bar_label(container, fontsize=10)

    axes[0].set_title('Bar Chart', fontsize=13, fontweight='semibold')
    axes[0].set_xlabel(column, fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].tick_params(axis='x', rotation=45)

    # --- PIE CHART ---
    # Check if column is numerical (pie charts work best for categorical data)
    if df[column].nunique() <= 10:  # Avoid too many slices
        df[column].value_counts().plot.pie(
            autopct='%1.1f%%',
            startangle=90,
            ax=axes[1],
            colors=pie_palette,
            textprops={'fontsize': 10}
        )
        axes[1].set_ylabel('')
        axes[1].set_title('Pie Chart', fontsize=13, fontweight='semibold')
    else:
        # Hide pie chart if too many categories
        axes[1].text(0.5, 0.5, 'Too many categories\nfor pie chart', 
                    ha='center', va='center', fontsize=12)
        axes[1].axis('off')

    # Adjust layout
    plt.tight_layout(pad=3)
    plt.subplots_adjust(top=0.85)
    plt.show()


# In[69]:


# Custom pastel and husl palettes
bar_palette = sns.color_palette("pastel")
pie_palette = sns.color_palette("husl", n_colors=10)

for i in df.columns:
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(20, 6), dpi=130)
    fig.suptitle(i, fontsize=16, fontweight='bold', color='#333')

    # Bar Chart
    sns.countplot(data=df, x=i, ax=axes[0], hue='Sex', palette=bar_palette,width=0.4, legend=False)
    for bar in axes[0].containers:
        axes[0].bar_label(bar, fontsize=10)

    axes[0].set_title('Bar Chart', fontsize=13, fontweight='semibold')
    axes[0].set_xlabel(i, fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].tick_params(axis='x', rotation=45)

    # Pie Chart
    df[i].value_counts().plot.pie(autopct='%1.1f%%',startangle=90,ax=axes[1], colors=pie_palette, textprops={'fontsize': 10})
    axes[1].set_ylabel('')
    axes[1].set_title('Pie Chart', fontsize=13, fontweight='semibold')

    # Layout styling
    plt.tight_layout(pad=3)
    plt.subplots_adjust(top=0.85)
    plt.show()


# In[70]:


from sklearn.utils import resample


# In[71]:


"""# Undersampling

df_majority = df[df.heart_disease == 0]
df_minority = df[df.heart_disease == 1]

df_major_down = resample(
    df_majority,
    replace=False,
    n_samples=len(df_minority),
    random_state=42
)

df_balanced = pd.concat([df_major_down, df_minority])

df_balanced['heart_disease'].value_counts()"""


# In[72]:


# Oversampling
from sklearn.utils import resample
df_majority = df[df.heart_disease == 0]
df_minority = df[df.heart_disease == 1]

df_minor_up = resample(
    df_minority,
    replace=True,
    n_samples=len(df_majority),
    random_state=42
)
df_balanced_2 = pd.concat([df_majority, df_minor_up])

df_balanced_2['heart_disease'].value_counts()


# In[73]:


"""from imblearn.over_sampling import SMOTE

X = df.drop('heart_disease',axis=1)
y = df['heart_disease']
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# 1. Dividir datos en entrenamiento y prueba (¡ANTES de aplicar SMOTE!)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Aplicar SMOTE solo a los datos de entrenamiento
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 3. Verificar el balance
print("Distribución original:", y_train.value_counts())
print("Distribución después de SMOTE:", y_train_resampled.value_counts())
Nunca apliques SMOTE antes de dividir los datos (generaría "fugas de datos").

Funciona mejor con variables numéricas. Si tienes categóricas, considera SMOTENC."""


# In[74]:


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np

# 1. Feature Scaling (Critical for logistic regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Model Training with balanced classes and regularization
model = LogisticRegression(
    class_weight='balanced',  # Adjusts for imbalanced classes
    max_iter=1000,            # Ensure convergence
    penalty='l2',             # Regularization to prevent overfitting
    C=1.0,                    # Inverse of regularization strength
    solver='liblinear',       # Works well for small-to-medium datasets
    random_state=42
)
model.fit(X_train_scaled, y_train)

# 3. Probability predictions and custom threshold
probs = model.predict_proba(X_test_scaled)[:, 1]
y_pred = (probs > 0.3).astype(int)  # Lower threshold for higher recall

# 4. Comprehensive Evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nROC-AUC Score:", roc_auc_score(y_test, probs))

# 5. Threshold Analysis (Optional)
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

precision, recall, thresholds = precision_recall_curve(y_test, probs)
plt.figure(figsize=(8, 4))
plt.plot(thresholds, precision[:-1], label='Precision')
plt.plot(thresholds, recall[:-1], label='Recall')
plt.axvline(0.3, color='red', linestyle='--', label='Your Threshold (0.3)')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.legend()
plt.title('Precision-Recall Tradeoff')
plt.show()


# In[75]:


from sklearn.metrics import accuracy_score

# Calculate accuracy with your custom threshold (0.3)
custom_threshold_accuracy = accuracy_score(y_test, y_pred)

# Compare with default threshold (0.5) accuracy
y_pred_default = (probs > 0.5).astype(int)
default_threshold_accuracy = accuracy_score(y_test, y_pred_default)

print(f"Accuracy (Threshold=0.3): {custom_threshold_accuracy:.4f}")
print(f"Accuracy (Threshold=0.5): {default_threshold_accuracy:.4f}")


# In[76]:


thresholds = np.linspace(0, 1, 50)
accuracies = [accuracy_score(y_test, (probs > t).astype(int)) for t in thresholds]

plt.plot(thresholds, accuracies)
plt.axvline(0.3, color='red', ls='--', label='Your Threshold')
plt.xlabel('Threshold')
plt.ylabel('Accuracy')
plt.legend();


# Age:
# The average age in the dataset is 54 years
# The oldest is 77 years, and the youngest is 29 years old.
# Resting blood pressure:
# The average is 131 , max = 180 and min = 94
# Cholesterol:
# The average registered cholestrol level is 245.34
# Maximum level is 417 and the minimum level is 126.
# Note: According to researches, a healthy cholesterol level is <200mg/dl and usually high level of cholesterol is associated with heart disease.
# St_depression:
# The average value of st_dpression is 0.999. Max is 4.4 and the minimum is 0.
# Max heart rate achieved:
# The average max heart rate registered is 149.93 bpm. The Maximum and the minumum are 202 and 88 bpm respectively.

# In[77]:


import numpy as np
from sklearn.preprocessing import StandardScaler
# 2. Inicializar el scaler y ajustarlo SOLO a X_train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Ajuste + transformación en train

# 3. Aplicar la misma transformación a X_test (sin fit, para evitar leakage)
X_test_scaled = scaler.transform(X_test)

# Opcional: Convertir a DataFrame (para legibilidad)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)


# In[78]:


X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)


# In[79]:


from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test, probs)
# Find threshold that maximizes F1-score or other metric
optimal_idx = np.argmax(2 * (precision * recall) / (precision + recall))
optimal_threshold = thresholds[optimal_idx]
y_pred_optimal = (probs > optimal_threshold).astype(int)


# In[80]:


from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# For imbalanced classes:
from sklearn.metrics import roc_auc_score
print("ROC-AUC:", roc_auc_score(y_test, probs))


# In[81]:


model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    penalty='l2',        # or 'l1' for feature selection
    C=0.1,               # inverse of regularization strength
    solver='liblinear'    # works well with L1/L2
)


# In[82]:


# Check correlation matrix before training
corr_matrix = X_train.corr().abs()


# In[83]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:,1]
y_pred = (probs > 0.3).astype(int)  # menor umbral para capturar más positivos


# In[84]:


"""# 2. MinMax
mms = MinMaxScaler()
X_train_mm = mms.fit_transform(X_train)
X_test_mm  = mms.transform(X_test)
# 3. Robust
rs = RobustScaler()
X_train_rs = rs.fit_transform(X_train)
X_test_rs  = rs.transform(X_test)
# 4. Log1p
X_train_log = np.log1p(X_train])
X_test_log  = np.log1p(X_test)
# 5. Sqrt
X_train_sqrt = np.sqrt(X_train)
X_test_sqrt  = np.sqrt(X_test)
# 6. Power (Yeo-Johnson)
pt = PowerTransformer(method='yeo-johnson')
X_train_pt = pt.fit_transform(X_train)
X_test_pt  = pt.transform(X_test)
# 7. Normalizer (por fila)
nm = Normalizer()
X_train_nm = nm.fit_transform(X_train)
X_test_nm  = nm.transform(X_test)"""


# In[85]:


# librerías
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib


# In[86]:


# 8) Guardar datos transformados
import numpy as np

np.savetxt('X_train_transformed.csv', X_train, delimiter=',')
np.savetxt('X_test_transformed.csv', X_test, delimiter=',')
np.savetxt('y_train.csv', y_train, delimiter=',')


# In[87]:


# 9) Corregir desbalance con SMOTE sobre el set seleccionado
sm = SMOTE(random_state=42)
X_train_sel, y_train_sel = sm.fit_resample(X_train, y_train)

len(X_train_sel), len(y_train_sel)


# In[88]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_auc_score

clf=LogisticRegression()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("Accuracy score {}".format(accuracy_score(y_test,y_pred)))
print("ROC AUC score {}".format(roc_auc_score(y_test,y_pred)))
#pd.DataFrame(data={"Y_Actual":y_test,"Y_Predict":y_pred})


# In[89]:


#Support Vector Machine Classifier
from sklearn.svm import SVC

clf=SVC(kernel='linear',gamma='scale')
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("Accuracy score {}".format(accuracy_score(y_test,y_pred)))
print("ROC AUC score {}".format(roc_auc_score(y_test,y_pred)))
#pd.DataFrame(data={"Y_Actual":y_test,"Y_Predict":y_pred})


# In[90]:


#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=60)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("Accuracy score {}".format(accuracy_score(y_test,y_pred)))
print("ROC AUC score {}".format(roc_auc_score(y_test,y_pred)))
#pd.DataFrame(data={"Y_Actual":y_test,"Y_Predict":y_pred})


# In[91]:


#KNN
from sklearn.neighbors import KNeighborsClassifier

clf=KNeighborsClassifier(n_neighbors=11)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("Accuracy score {}".format(accuracy_score(y_test,y_pred)))
print("ROC AUC score {}".format(roc_auc_score(y_test,y_pred)))
#pd.DataFrame(data={"Y_Actual":y_test,"Y_Predict":y_pred})


# In[92]:


#Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
clf=GradientBoostingClassifier(n_estimators=50,learning_rate=0.2)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("Accuracy score {}".format(accuracy_score(y_test,y_pred)))
print("ROC AUC score {}".format(roc_auc_score(y_test,y_pred)))
#pd.DataFrame(data={"Y_Actual":y_test,"Y_Predict":y_pred})


# In[93]:


#Naivye Bayes

from sklearn.naive_bayes import GaussianNB
clf=GaussianNB()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("Accuracy score {}".format(accuracy_score(y_test,y_pred)))
print("ROC AUC score {}".format(roc_auc_score(y_test,y_pred)))
#pd.DataFrame(data={"Y_Actual":y_test,"Y_Predict":y_pred})


# In[94]:


#fitting the Xgboost classifier on the training set
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
model= XGBClassifier(learning_rate= 0.1, max_depth=5, min_child_weight= 3, eval_metric='auc')
model.fit(X_train_scaled, y_train)


# In[95]:


#using the Xgboost model to predict the test set and checking it's accuracy

prediction= model.predict(X_test_scaled)

print(model.score(X_test_scaled, y_test))
print(classification_report(y_test, prediction))


# In[96]:


#fitting the training set on a random forest classifier 

classifier= RandomForestClassifier(max_features=4, max_depth=2, random_state=0)
classifier.fit(X_train_scaled, y_train)


# In[97]:


predict= classifier.predict(X_test)
print(classifier.score(X_train_scaled, y_train))
print(classifier.score(X_test_scaled, y_test))
print(classification_report(y_test, predict))


# In[98]:


#fitting the training set on a Kneighbors classifier 

Neighborhood= KNeighborsClassifier(n_neighbors=5).fit(X_train_scaled, y_train)


# In[99]:


#Using the Kneighbors model to predict the test set and checking the accuracy 

pred= Neighborhood.predict(X_test_scaled)
print(Neighborhood.score(X_train_scaled, y_train))
print(Neighborhood.score(X_test_scaled, y_test))
print(classification_report(y_test, pred))


# In[125]:


from sklearn.ensemble import GradientBoostingClassifier


# In[126]:


gradientB =GradientBoostingClassifier(n_estimators=100, max_depth =3)
gradientB.fit(X_train_scaled, y_train)


# In[127]:


yhat=gradientB.predict(X_test_scaled)
yhat


# In[128]:


from sklearn import metrics 
acc_gradient=metrics.accuracy_score(y_test,yhat)
acc_gradient


# In[129]:


from sklearn.tree import DecisionTreeClassifier


# In[130]:


deTree = DecisionTreeClassifier(criterion='entropy')


# In[131]:


deTree.fit(X_train_scaled,y_train)


# In[132]:


yhat=deTree.predict(X_test_scaled)
yhat


# In[133]:


from sklearn import metrics
acc_deTree=metrics.accuracy_score(y_test,yhat)
acc_deTree


# In[134]:


print(classification_report(y_test,yhat))


# In[135]:


from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print(confusion_matrix(y_test, yhat, labels=[1,0]))


# In[136]:


cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['output=1','output=0'],normalize= False,  title='Confusion matrix')


# In[137]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


# In[138]:


forestRandom= RandomForestClassifier(n_estimators=100 ,max_depth=3 , criterion="entropy")
forestRandom.fit(X_train_scaled, y_train)


# In[139]:


yhat=forestRandom.predict(X_test_scaled)
yhat[:5]


# In[140]:


acc_forest=metrics.accuracy_score(y_test,yhat)
acc_forest


# In[141]:


from sklearn import svm
sVm= svm.SVC(kernel='linear')


# In[142]:


sVm.fit(X_train_scaled, y_train)


# In[143]:


yhat = sVm.predict(X_test_scaled)
yhat [0:5]


# In[144]:


acc_sVm=metrics.accuracy_score(y_test,yhat)
acc_sVm


# In[145]:


from sklearn.linear_model import LogisticRegression
logisticR =LogisticRegression(C=0.01)
logisticR.fit(X_train_scaled,y_train)
logisticR


# In[146]:


yhat =logisticR.predict(X_test_scaled)
yhat


# In[147]:


acc_lr=metrics.accuracy_score(y_test,yhat)
acc_lr


# In[152]:


input_data = (59,3,3,200,223,2,3,100,0,2.2,2)

input_data_as_numpy_array = np.array(input_data)

input_data_reshape = input_data_as_numpy_array.reshape(1,-1)

std_data = scaler.transform(input_data_reshape)
print(std_data)

prediction = Neighborhood.predict(std_data)
print(prediction)

if (prediction[0] == 0):
   input_data = (63,1,3,145,233,1,0,150,0,2.3,0,0,1)

input_data_as_numpy_array = np.array(input_data)

input_data_reshape = input_data_as_numpy_array.reshape(1,-1)

std_data = scaler.transform(input_data_reshape)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
    print('Low likelihood of a heart attack')
else :
    print('Higher likelihood of a heart attack')




import pickle


# Guardar el modelo entrenado
nombre_archivo = 'attack_model.pkl'
with open(nombre_archivo, 'wb') as archivo:
    pickle.dump(Neighborhood, archivo)

print(f"Modelo guardado en {nombre_archivo}")














