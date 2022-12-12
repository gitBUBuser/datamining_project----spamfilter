import pandas as pd
import numpy as np
import os
import email
import email.policy


spam_folder = "spam"
ham_folder = "ham"

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")

def load_emails(spam = False):
    directory = ""
    if spam:
        directory = os.path.join(data_path, spam_folder)
    else:
        directory = os.path.join(data_path, ham_folder)

    


