#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import datetime as dt
from sklearn.model_selection import train_test_split

data = pd.read_csv('wildfires.csv') 
# In[2]:


#drop duplicates
data = data.drop_duplicates()
# converting date from julian
data['DATE'] = pd.to_datetime(data['DISCOVERY_DATE'] - pd.Timestamp(0).to_julian_date(), unit='D')
# Converting to date to month of year
data['MONTH'] = data['DATE'].dt.month
data['MONTH'] = data.MONTH.replace([1,2,3,4,5,6,7,8,9,10,11,12], ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
print(data['MONTH'])
data['DAY_OF_YEAR'] = data['DATE'].dt.dayofyear


# In[3]:


lst = ['DAY_OF_YEAR', 'MONTH', 'FIRE_YEAR', 'STATE',        'LATITUDE', 'LONGITUDE', 'FIRE_SIZE', 'STAT_CAUSE_DESCR']

data_orig = data.copy()
data = data[lst]
data.head()


# In[4]:


misc_fires = data[data['STAT_CAUSE_DESCR'] == "Miscellaneous"]
missing_fires = data[data['STAT_CAUSE_DESCR'].str.contains("missing", case= False)]

df = data[data['STAT_CAUSE_DESCR'] != "Miscellaneous"]

df = df[~df['STAT_CAUSE_DESCR'].str.contains("missing", case= False)]

#df = data.loc([(data['STAT_CAUSE_DESCR'] != "Miscellaneous") & (~data['STAT_CAUSE_DESCR'].str.contains("missing", case= False))])

df


# In[5]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(df.STAT_CAUSE_DESCR)
df['categorical_label'] = le.transform(df.STAT_CAUSE_DESCR)
df



# In[ ]:



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
#from sklearn.externals import joblib

#df = one_hot_df

factor_cause = pd.factorize(df['STAT_CAUSE_DESCR'])
df['cause'] = factor_cause[0]
cause_definitions = factor_cause[1]

# factor_state = pd.factorize(df['STATE'])
# df['state'] = factor_state[0]
# state_definitions = factor_state[1]

factor_MONTH = pd.factorize(df['MONTH'])
df['month'] = df.MONTH.replace(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], [1,2,3,4,5,6,7,8,9,10,11,12])

df_temp = df.drop(['STAT_CAUSE_DESCR', 'STATE', 'MONTH', 'categorical_label'], axis=1)

#df_temp = df.drop(['state', 'month'], axis=1)

#df_temp = df_temp.copy()[:10000]

df_temp = df_temp.sample(n=df_temp.shape[0]#100000
                         , random_state=42)

df_temp


# In[34]:


#Splitting the data into independent and dependent variables
rf_X = df_temp.drop(["cause"], axis=1).values
rf_y = df_temp['cause'].values
rf_X_train, rf_X_test, rf_y_train, rf_y_test = train_test_split(rf_X, rf_y, test_size = 0.25, random_state = 21)


# In[35]:


#scaler = StandardScaler()
# rf_X_train = scaler.fit_transform(rf_X_train)
# rf_X_test = scaler.transform(rf_X_test)

classifier = RandomForestClassifier(max_depth = 24, n_estimators = 100, criterion = 'entropy', random_state = 42)
classifier.fit(rf_X_train, rf_y_train)


# In[36]:


# import pickle

# pickle.dump(classifier, open('model.pkl','wb'))

# model = pickle.load(open('model.pkl','rb'))


# In[ ]:



# In[ ]:


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = classifier
#model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    flt_features = [float(x) for x in request.form.values()]
    final_features = [np.array(flt_features)]
    prediction = model.predict(final_features)
    
    cause_def = ['Lightning','Debris Burning','Campfire','Equipment Use','Arson','Children','Railroad','Smoking','Powerline','Structure','Fireworks']

    output_num = prediction[0]
    output = cause_def[output_num]
    

    return render_template('index.html', prediction_text='Fire cause should be {}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    cause_def = ['Lightning','Debris Burning','Campfire','Equipment Use','Arson','Children','Railroad','Smoking','Powerline','Structure','Fireworks']

    output_num = prediction[0]
    output = cause_def[output_num]
    
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)


# In[ ]:




