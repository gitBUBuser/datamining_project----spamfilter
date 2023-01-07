from pandas_ods_reader import read_ods
import numpy as np
import os
import email
from email import parser as e_parser
from email import message as e_message
import re 
from pyexcel_ods3 import save_data
from collections import OrderedDict
import random
import io

# class for stroing email messages in an intuitive and understandable manner
class SimplifiedMessage():
    def __init__(self, email_message):
        self.pure_message = email_message
        self.text_body = ""

        reg_emails = re.compile('([A-z]+[@]+\w+[.]+\w+)')

        if email_message.is_multipart():
            
            for part in email_message.walk():
                if part.get_content_type() == 'text/plain':
                    try:
                        self.text_body += part.get_content() + ' '
                    except:
                        return None
        else: 
            try:
                self.text_body = email_message.get_content()
            except:
                pass
        
        self.sender = reg_emails.findall(email_message.get("From", failobj="None"))
        self.receivers = reg_emails.findall(email_message.get("to", failobj="None"))
        self.ccs = reg_emails.findall(email_message.get("cc", failobj="None"))

    def print(self):
        print(f"sender: {self.sender}")

        print(f"receivers: {self.receivers}")
        print(f"CCs: {self.ccs}")
        print()
        print("___________________________________________________")
        print()
        print(self.text_body)
        
# class for stroring, removing information about email attributes
class EmailInfo():
    def __init__(self, spam, text_body, id):
        if text_body == None:
            self.info = {
                "id": id,
                "spam": spam,
                "raw": "none",
            }

        else:
            self.info = {
                "id": id,
                "spam": spam,
                "raw": text_body,
            }
        
    def set_attributes(self, args_dict):
        for key, value in args_dict.items():
            self.set_attribute(key, value)    

    def set_attribute(self, name, value):
        self.info[name] = value

    def get_info(self):
        return self.info

    def set_id(self, new_id):
        self.set_attribute("id", new_id)

    def get(self, attribute):
        return self.info[attribute]

    def all_features(self):
        feature_vector = self.info.copy()
        feature_vector.pop("id")
        feature_vector.pop("spam")
        feature_vector.pop("raw")
        return [self.get(key) for key in feature_vector.keys()]
    
    def get_attributes(self,attributes):
        if attributes == ["all"]:
            return self.all_features()
        else:
            return [self.get(key) for key in attributes]

    def spam_class(self):
        return self.get("spam")
            

# class for storing a list of email attributes
class EmailAttributeList():
    def __init__(self):
        self.length = 0
        self.emails = []
    
    def add_emails(self, y, emails):
        for i in range(len(emails)):
            if type(y) is int:
                self.add_email(y, emails[i])
            else:
                self.add_email(y[i], emails[i])

    def get_emails(self):
        return self.emails        
    
    def add_email(self, spam, email):
        self.emails.append(EmailInfo(spam, email.text_body, self.length))
        self.length += 1
        
    def add_email_with_args(self, args):
        email = EmailInfo(0, None, 0)
        email.set_attributes(args)
        self.emails.append(email)

    def find_email_by_id(self, id):
        for email in self.emails:
            if email.id == id:
                return email

    def delete_email_by_id(self, id):
        self.emails.pop(self.find_email_by_id(id))

    def get_email_keys(self):
        return self.get_emails()[0].get_info().keys()

    def save(self, save_raw = False):
        keys = self.get_email_keys()
        emails = self.get_emails()

        save_dict = {}
        
        if not save_raw:
            for email in emails:
                email.get_info().pop("raw")

        for key in keys:
            save_dict[key] = []
        
        for email in emails:
            for key in keys:
                save_dict[key].append(email.get(key))

        save_list = []
        for key in save_dict.keys():
            save_list.append(key)
            save_list.append(save_dict[key])
        
        save = OrderedDict()
        rows = []
        for key, value in save_dict.items():
            rows.append([key] + value)
        columns = np.array(rows).transpose().tolist()
        save.update({"Sheet 1": columns})
        save_data("AttributeData.ods", save)

    def load(self, path):
        file = read_ods(path)
        print(file)
        print(file.shape)
        
        emails = []
   
        for row in range(0, file.shape[0]):
            attributes = {}
        
            for column in file.columns:
                attributes[column] = file.at[row, column]            
            self.add_email_with_args(attributes)

    def restrict_to_amount_per_class(self, amount_per_class):
        spam_emails = []
        ham_emails = []

        for email in self.get_emails():
            if email.spam_class() == '1':
                spam_emails.append(email)
            else:
                ham_emails.append(email)
                
        spam_emails = random.sample(spam_emails, amount_per_class)
        ham_emails = random.sample(ham_emails, amount_per_class)
        
        self.emails = spam_emails + ham_emails


    # attributes = all by defualt, one can choose which attributes one wishes to get from the feature vector
    def to_feature_vector(self, attributes = ["all"], amount_per_class = None):
        y = []
        X = []
        
        if  amount_per_class == None:
            for email in self.get_emails():
                y.append(email.spam_class())
                X.append(email.get_attributes(attributes))
        else:
            spam_emails = []
            ham_emails = []

            for email in self.get_emails():
                if email.spam_class() == '1':
                    spam_emails.append(email)
                else:
                    ham_emails.append(email)
                    
            spam_emails = random.sample(spam_emails, amount_per_class)
            ham_emails = random.sample(ham_emails, amount_per_class)
            emails = spam_emails + ham_emails
            for email in emails:
                y.append(email.spam_class())
                X.append(email.get_attributes(attributes))

        return np.array(X, dtype=float), np.array(y, dtype=int)
