# File Path to Data Sourced from https://www.consumerfinance.gov/data-research/consumer-complaints/#download-the-data
config_complaints_file = 'D:/complaint_data/complaints.csv'
config_complaints_file_encoding = 'cp850'
config_complaints_narrative_column = 'Consumer complaint narrative'
config_model_folder = 'D:/complaint_labeling_neural_net/models/'
config_issue_model_save_name = f'{config_model_folder}lstm_issue_classifier.hdf5'
config_vectorizer_folder = 'D:/complaint_labeling_neural_net/vectorizers/'
config_proc_data_dir = 'D:/complaint_labeling_neural_net/processed_data/'

# Product-Issue File Names
config_train_x_save_name_product_issue = f'{config_proc_data_dir}train_sequences_product_issue.npy'
config_test_x_save_name_product_issue = f'{config_proc_data_dir}test_sequences_product_issue.npy'
config_train_y_save_name_product_issue = f'{config_proc_data_dir}train_y_product_issue.npy'
config_test_y_save_name_product_issue = f'{config_proc_data_dir}test_y_product_issue.npy'
config_tokenizer_save_name_product_issue = f'{config_vectorizer_folder}keras_tokenizer_product_issue.pkl'
config_product_issue_model_save_name = f'{config_model_folder}lstm_product_issue_classifier.hdf5'



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

# Additional Stopwords
config_custom_stopwords = ['x' * i for i in range(1,21)]


# Contractions
config_contraction_dict = {"ain't":"am not",
                            "aren't":"are not",
                            "can't":"cannot",
                            "could've":"could have",
                            "couldn't":"could not",
                            "couldn't've":"could not have",
                            "daren't":"dare not",
                            "daresn't":"dare not",
                            "dasn't":"dare not",
                            "didn't":"did not",
                            "doesn't":"does not",
                            "don't":"do not",
                            "dunno":"do not know",
                            "everybody's":"everybody is",
                            "everyone's":"everyone is",
                            "finna":"going to",
                            "g'day":"good day",
                            "gimme":"give me",
                            "gonna":"going to",
                            "gotta":"got to",
                            "hadn't":"had not",
                            "had've":"had have",
                            "hasn't":"has not",
                            "haven't":"have not",
                            "he'd":"he would",
                            "he'll":"he will",
                            "he's":"he is",
                            "he've":"he have",
                            "how'd":"how did",
                            "how'll":"how will",
                            "how're":"how are",
                            "how's":"how is",
                            "I'd":"I had",
                            "I'd've":"I would have",
                            "I'll":"I will",
                            "I'm":"I am",
                            "innit":"is it not",
                            "I've":"I have",
                            "isn't":"is not",
                            "it'd":"it would",
                            "it'll":"it will",
                            "it's":"it is",
                            "let's":"let us",
                            "ma'am":"madam",
                            "mayn't":"may not",
                            "may've":"may have",
                            "methinks":"me thinks",
                            "mightn't":"might not",
                            "might've":"might have",
                            "mustn't":"must not",
                            "mustn't've":"must not have",
                            "must've":"must have",
                            "needn't":"need not",
                            "nal":"and all",
                            "ne'er":"never",
                            "o'clock":"of the clock",
                            "o'er":"over",
                            "ol'":"old",
                            "oughtn't":"ought not",
                            "shalln't":"shall not",
                            "shan't":"shall not",
                            "she'd":"she would",
                            "she'll":"she will",
                            "she's":"she is",
                            "should've":"should have",
                            "shouldn't":"should not",
                            "shouldn't've":"should not have",
                            "somebody's":"somebody is",
                            "someone's":"someone is",
                            "something's":"something is",
                            "so're":"so are",
                            "that'll":"that will",
                            "that're":"that are",
                            "that's":"that is",
                            "that'd":"that would",
                            "there'd":"there would",
                            "there'll":"there will",
                            "there're":"there are",
                            "there's":"there is",
                            "these're":"these are",
                            "these've":"these have",
                            "they'd":"they would",
                            "they'll":"they will",
                            "they're":"they are",
                            "they've":"they have",
                            "this's":"this is",
                            "those're":"those are",
                            "those've":"those have",
                            "'tis":"it is",
                            "to've":"to have",
                            "'twas":"it was",
                            "wanna":"want to",
                            "wasn't":"was not",
                            "we'd":"we would",
                            "we'd've":"we would have",
                            "we'll":"we will",
                            "we're":"we are",
                            "we've":"we have",
                            "weren't":"were not",
                            "what'd":"what did",
                            "what'll":"what will",
                            "what're":"what are",
                            "what's":"what is",
                            "what've":"what have",
                            "when's":"when is",
                            "where'd":"where did",
                            "where'll":"where will",
                            "where're":"where are",
                            "where's":"where is",
                            "where've":"where have",
                            "which'd":"which had",
                            "which'll":"which will",
                            "which're":"which are",
                            "which's":"which is",
                            "which've":"which have",
                            "who'd":"who would",
                            "who'd've":"who would have",
                            "who'll":"who will",
                            "who're":"who are",
                            "who's":"who is",
                            "who've":"who have",
                            "why'd":"why did",
                            "why're":"why are",
                            "why's":"why is",
                            "willn't":"will not",
                            "won't":"will not",
                            "wonnot":"will not",
                            "would've":"would have",
                            "wouldn't":"would not",
                            "wouldn't've":"would not have",
                            "y'all":"you all",
                            "y'all'd've":"you all would have",
                            "y'all'd'n've":"you all would not have",
                            "y'all're":"you all are",
                            "you'd":"you would",
                            "you'll":"you will",
                            "you're":"you are",
                            "you've":"you have"}










