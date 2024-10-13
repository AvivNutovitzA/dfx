import numpy as np
import pandas as pd


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


read_prefix = '../../'
write_prefix = read_prefix+'results/datasets/'
model_name = '/DNN'
all_data_set_names = ['stroke', 'fetal_health', 'bank_churners', 'hr_employee_attrition',
                      'cervical_cancer', 'mobile_price', 'churn_modelling',
                      'company_bankruptcy_prediction', 'airline_passenger_satisfaction',
                      'banking_marketing_targets']

for data_set_name in all_data_set_names:
    file_name1 = write_prefix+data_set_name+model_name+'_model_dfx_importance_as_df.csv'
    file_name2 = write_prefix+data_set_name+model_name+'_model_shap_importance_as_df.csv'

    df1 = pd.read_csv(file_name1)
    df2 = pd.read_csv(file_name2)

    res_inter = []
    res_comp = []
    for index, (df1_row, df2_row) in enumerate(zip(df1.iterrows(), df2.iterrows())):
        l1 = df1.iloc[:index+1]['feature_name'].tolist()
        l2 = df2.iloc[:index+1]['feature_name'].tolist()
        res_inter.append(np.round(len(intersection(l1, l2)) / len(l1), 2))
        res_comp.append(np.round((index + 1) / df1.shape[0], 2))
    res_df = pd.DataFrame({'inter': res_inter, "comp": res_comp})

    inter = []
    comp = []
    if res_df.shape[0] >= 10:
        for i in range(1, 11):
            inter.append(float(res_df[res_df['comp'] <= i/10]['inter'].tail(1)))
            comp.append(i/10)

        res_df1 = pd.DataFrame({'inter': inter, "comp": comp})
        res_df1.to_csv(write_prefix+data_set_name + model_name+'_dfx_shap_inter_comp_prop.csv', index=False)
    else:
        res_df.to_csv(write_prefix + data_set_name + model_name+'_dfx_shap_inter_comp_prop.csv', index=False)



