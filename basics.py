from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('lc_loan.pkl','rb'))

app=Flask(__name__)

@app.route('/')
def man():
    return render_template('front_end.html')


@app.route('/predict', methods=['POST'])
def predict():
    data1 = request.form['gender']
    data2 = request.form['married']
    data3 = request.form['dependents']
    data4 = request.form['education']
    data5 = request.form['self_employed']
    data6 = request.form['applicantincome']
    data7 = request.form['coapplicantincome']
    data8 = request.form['loanamount']
    data9 = request.form['loan_amount_term']
    data10 = request.form['credit_history']
    data11 = request.form['property_area']
    arr = np.array([data1, data2, data3, data4 ,data5 ,data6 ,data7 ,data8 ,data9 ,data10 ,data11])
    to_predict_list = list(map(int, arr))
    pred = model.predict([to_predict_list])

    return render_template('result.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)
