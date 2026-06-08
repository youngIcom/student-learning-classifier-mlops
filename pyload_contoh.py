import json
import pandas as pd

df = pd.read_csv("/home/yesaya/Documents/Pijak Program/proyek_akhir_membangun_sistem_machine_learning/Eksperimen_SML_Yesaya/open_UL_analysis_preprocessing/student_cleaned.csv")
features = json.load(open("Workflow_CI_Yesaya/artifacts/feature_columns.json"))
sample = df[features].iloc[0].to_dict()
payload = {"features": {k: float(v) for k, v in sample.items()}}
print(json.dumps(payload))