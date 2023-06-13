from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application



@app.route('/', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        data = CustomData(
            Item_Fat_Content=request.form.get('Item_Fat_Content'),
            Item_Type=request.form.get('Item_Type'),
            Outlet_Location_Type=request.form.get('Outlet_Location_Type'),
            Outlet_Type=request.form.get('Outlet_Type'),
            Outlet_Size=request.form.get('Outlet_Size'),
            Item_Visibility=float(request.form.get('Item_Visibility')),
            Item_MRP=int(request.form.get('Item_MRP')),
            Age_Outlet=int(request.form.get('Age_Outlet')),
            Item_Weight=float(request.form.get('Item_Weight'))

        )

        pred_df = data.get_data_as_data_frame()
        # print(pred_df)
        # print("Before Prediction")

        predict_pipeline = PredictPipeline()
        # print("Mid Prediction")
        output = predict_pipeline.predict(pred_df)
        # print("after Prediction")
        msg = f"Sale of that item will be: {round(output[0],2)}"
        return render_template('index.html',results=msg)


if __name__ == "__main__":
    app.run(host="0.0.0.0")