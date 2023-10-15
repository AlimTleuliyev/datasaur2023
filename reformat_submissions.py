import pandas as pd

b0 = pd.read_csv('efficientnet_b0_submission.csv')

b0['class'] = b0['class'].replace(2, 1)
b0['class'] = b0['class'].replace(3, 1)
b0['class'] = b0['class'].replace(4, 1)

b0.to_csv('efficientnet_b0_submission_final.csv', index=False)

b1 = pd.read_csv('efficientnet_b1_submission.csv')

b1['class'] = b1['class'].replace(2, 1)
b1['class'] = b1['class'].replace(3, 1)
b1['class'] = b1['class'].replace(4, 1)

b1.to_csv('efficientnet_b1_submission_final.csv', index=False)

b2 = pd.read_csv('efficientnet_b2_submission.csv')

b2['class'] = b2['class'].replace(2, 1)
b2['class'] = b2['class'].replace(3, 1)
b2['class'] = b2['class'].replace(4, 1)

b2.to_csv('efficientnet_b2_submission_final.csv', index=False)
