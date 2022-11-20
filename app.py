from flask import Flask,jsonify,request,render_template
from project_app.utils import Diabetes
# from sklearn import neighbors
import config

app=Flask(__name__)
@app.route("/")
def home():
    print("This is Iris dataset model")
    return render_template("home.html")

@app.route('/predict', methods=['POST','GEt'])
def predict():
    if request.method == "POST":
        data=request.form
        print(data)
        Glucose= eval(data["Glucose"])
        BloodPressure=eval(data["BloodPressure"])
        SkinThickness=eval(data["SkinThickness"])
        Insulin=eval(data["Insulin"])
        BMI=eval(data["BMI"])
        DiabetesPedigreeFunction=eval(data["DiabetesPedigreeFunction"])
        Age=eval(data["Age"])

        obj=Diabetes(Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age)
        result=obj.predict_diabetes()
        print("Result is: ",result)
        return render_template("after.html", data=result)
        # return render_template("after.html", data=result)

if __name__=="__main__":
    app.run(host="0.0.0.0",port=config.PORT_NUMBER,debug=True)
