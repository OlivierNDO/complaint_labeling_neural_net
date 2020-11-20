# File Path to Data Sourced from https://www.consumerfinance.gov/data-research/consumer-complaints/#download-the-data
config_complaints_file = 'D:/complaint_data/complaints.csv'
config_complaints_file_encoding = 'cp850'
config_complaints_narrative_column = 'Consumer complaint narrative'
config_model_folder = 'D:/complaint_labeling_neural_net/models/'
config_issue_model_save_name = f'{config_model_folder}lstm_issue_classifier.hdf5'
config_vectorizer_folder = 'D:/complaint_labeling_neural_net/vectorizers/'
config_proc_data_dir = 'D:/complaint_labeling_neural_net/processed_data/'

# Product File Names
config_train_x_save_name_product = f'{config_proc_data_dir}train_sequences_product.npy'
config_test_x_save_name_product = f'{config_proc_data_dir}test_sequences_product.npy'
config_train_y_save_name_product = f'{config_proc_data_dir}train_y_product.npy'
config_test_y_save_name_product = f'{config_proc_data_dir}test_y_product.npy'
config_tokenizer_save_name_product = f'{config_vectorizer_folder}keras_tokenizer_product.pkl'
config_product_model_save_name = f'{config_model_folder}lstm_product_classifier.hdf5'

# Checking File Names
config_train_x_save_name_checking = f'{config_proc_data_dir}train_sequences_checking.npy'
config_test_x_save_name_checking = f'{config_proc_data_dir}test_sequences_checking.npy'
config_train_y_save_name_checking = f'{config_proc_data_dir}train_y_checking.npy'
config_test_y_save_name_checking = f'{config_proc_data_dir}test_y_checking.npy'
config_tokenizer_save_name_checking = f'{config_vectorizer_folder}keras_tokenizer_checking.pkl'
config_checking_model_save_name = f'{config_model_folder}lstm_checking_classifier.hdf5'

# Card File Names
config_train_x_save_name_card = f'{config_proc_data_dir}train_sequences_card.npy'
config_test_x_save_name_card = f'{config_proc_data_dir}test_sequences_card.npy'
config_train_y_save_name_card = f'{config_proc_data_dir}train_y_card.npy'
config_test_y_save_name_card = f'{config_proc_data_dir}test_y_card.npy'
config_tokenizer_save_name_card = f'{config_vectorizer_folder}keras_tokenizer_card.pkl'
config_card_model_save_name = f'{config_model_folder}lstm_card_classifier.hdf5'


# Credit Reporting File Names
config_train_x_save_name_cr = f'{config_proc_data_dir}train_sequences_cr.npy'
config_test_x_save_name_cr = f'{config_proc_data_dir}test_sequences_cr.npy'
config_train_y_save_name_cr = f'{config_proc_data_dir}train_y_cr.npy'
config_test_y_save_name_cr = f'{config_proc_data_dir}test_y_cr.npy'
config_tokenizer_save_name_cr = f'{config_vectorizer_folder}keras_tokenizer_cr.pkl'
config_cr_model_save_name = f'{config_model_folder}lstm_cr_classifier.hdf5'

# Debt Collection File Names
config_train_x_save_name_dc = f'{config_proc_data_dir}train_sequences_dc.npy'
config_test_x_save_name_dc = f'{config_proc_data_dir}test_sequences_dc.npy'
config_train_y_save_name_dc = f'{config_proc_data_dir}train_y_dc.npy'
config_test_y_save_name_dc = f'{config_proc_data_dir}test_y_dc.npy'
config_tokenizer_save_name_dc = f'{config_vectorizer_folder}keras_tokenizer_dc.pkl'
config_dc_model_save_name = f'{config_model_folder}lstm_dc_classifier.hdf5'

# Student Loan File Names
config_train_x_save_name_sl = f'{config_proc_data_dir}train_sequences_sl.npy'
config_test_x_save_name_sl = f'{config_proc_data_dir}test_sequences_sl.npy'
config_train_y_save_name_sl = f'{config_proc_data_dir}train_y_sl.npy'
config_test_y_save_name_sl = f'{config_proc_data_dir}test_y_sl.npy'
config_tokenizer_save_name_sl = f'{config_vectorizer_folder}keras_tokenizer_sl.pkl'
config_sl_model_save_name = f'{config_model_folder}lstm_sl_classifier.hdf5'

# Data Configuration
config_use_products = ['Credit reporting, credit repair services, or other personal consumer reports',
                       'Credit card or prepaid card',
                       'Debt collection',
                       'Checking or savings account',
                       'Student loan']

# Mapping of 'Product | Issue | Sub-issue' to Complaint Category
config_product_issue_list = ["Checking or savings account | Managing an account",
                             "Checking or savings account | Closing an account",
                             "Checking or savings account | Opening an account",
                             "Checking or savings account | Problem with a lender or other company charging your account",
                             "Checking or savings account | Problem caused by your funds being low",
                             "Credit card or prepaid card | Problem with a purchase shown on your statement",
                             "Credit card or prepaid card | Other features, terms, or problems",
                             "Credit card or prepaid card | Fees or interest",
                             "Credit card or prepaid card | Problem when making payments",
                             "Credit card or prepaid card | Getting a credit card",
                             "Credit card or prepaid card | Closing your account",
                             "Credit card or prepaid card | Advertising and marketing, including promotional offers",
                             "Credit reporting, credit repair services, or other personal consumer reports | Incorrect information on your report",
                             "Credit reporting, credit repair services, or other personal consumer reports | Problem with a credit reporting company's investigation into an existing problem",
                             "Credit reporting, credit repair services, or other personal consumer reports | Improper use of your report",
                             "Debt collection | Attempts to collect debt not owed",
                             "Debt collection | Cont'd attempts collect debt not owed",
                             "Debt collection | Communication tactics",
                             "Debt collection | Written notification about debt",
                             "Debt collection | False statements or representation",
                             "Debt collection | Took or threatened to take negative or legal action",
                             "Debt collection | Disclosure verification of debt",
                             "Student loan | Dealing with your lender or servicer", 
                             "Student loan | Dealing with my lender or servicer",
                             "Student loan | Struggling to repay your loan",
                             "Student loan | Can't repay my loan"]













