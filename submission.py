final_predictions = model.predict(X_test_official) # X_test_official is your cleaned test.csv

import pandas as pd

submission_df = pd.DataFrame({
    'Id': df_test['Id'],  
    'Predicted_Outcome': final_predictions
})

submission_df.to_csv('my_final_submission.csv', index=False)
print("Done! Your results are saved in 'my_final_submission.csv'")