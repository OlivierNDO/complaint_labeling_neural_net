# File Path to Data Sourced from https://www.consumerfinance.gov/data-research/consumer-complaints/#download-the-data
config_complaints_file = 'D:/complaint_data/complaints.csv'
config_complaints_file_encoding = 'cp850'
config_complaints_narrative_column = 'Consumer complaint narrative'
config_model_folder = 'D:/complaint_labeling_neural_net/models/'
config_product_model_save_name = f'{config_model_folder}/lstm_product_classifier.hdf5'
config_vectorizer_folder = 'D:/complaint_labeling_neural_net/vectorizers/'


# Data Configuration
config_product_dict = {'Debt collection': 'Debt collection',
 'Credit reporting, credit repair services, or other personal consumer reports': 'Credit reporting',
 'Money transfer, virtual currency, or money service': 'Money transfers or virtual currency',
 'Mortgage': 'Mortgage',
 'Student loan': 'Student loan',
 'Vehicle loan or lease': 'Vehicle loan or lease',
 'Credit card or prepaid card': 'Card',
 'Checking or savings account': 'Checking or savings account',
 'Credit card': 'Card',
 'Payday loan, title loan, or personal loan': 'Consumer Loan',
 'Consumer Loan': 'Consumer Loan',
 'Payday loan': 'Consumer Loan',
 'Credit reporting': 'Credit reporting',
 'Prepaid card': 'Card',
 'Other financial service': 'Other financial service',
 'Bank account or service': 'Other financial service',
 'Money transfers': 'Money transfers or virtual currency',
 'Virtual currency': 'Money transfers or virtual currency'}

