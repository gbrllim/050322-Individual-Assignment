#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask
app = Flask(__name__)


# In[2]:


from flask import request, render_template


# In[3]:


import joblib


# In[4]:


@app.route("/",methods=["GET","POST"])
def index():
    if request.method == "POST":
        income = request.form.get("income")
        age = request.form.get("age")
        loan = request.form.get("loan")
        income = float(income)
        age = float(age)
        loan = float(loan)

        model1 = joblib.load("CCD_Reg")
        pred1 = model1.predict([[income,age,loan]])
        s1 = "1) From the Logistic Regession model: " + str(pred1[0])
        
        model2 = joblib.load("CCD_DT")
        pred2 = model2.predict([[income,age,loan]])
        s2 = "2) From the Decision Tree model: " + str(pred2[0])
        
        model3 = joblib.load("CCD_RF")
        pred3 = model3.predict([[income,age,loan]])
        s3 = "3) From the Random Forest model: " + str(pred3[0])
        
        model4 = joblib.load("CCD_GB")
        pred4 = model4.predict([[income,age,loan]])
        s4 = "4) From the XGBoost model: " + str(pred4[0])
        
        model5 = joblib.load("CCD_NN")
        pred5 = model5.predict([[income,age,loan]])
        s5 = "5) From the Neural Network model: " + str(pred5[0])
        
        return(render_template("index.html",result1=s1,result2=s2,result3=s3,result4=s4,result5=s5))
    else:
        return(render_template("index.html",result1="No Input",result2="No Input",result3="No Input",result4="No Input",result5="No Input"))


# In[ ]:


if __name__=="__main__":
    app.run()


# In[ ]:




