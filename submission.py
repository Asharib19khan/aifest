final_predictions = model.predict(X_test_official)

import pandas as pd

submission_df = pd.DataFrame({
    'Id': df_test['Id'],  
    'Predicted_Outcome': final_predictions
})

submission_df.to_csv('my_final_submission.csv', index=False)
print("Saved in 'my_final_submission.csv'")