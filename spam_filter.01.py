import pandas as pd
import numpy as np
import os
import email
from email import parser as e_parser
from email import message as e_message
import email.policy
import sklearn as sk
import re 
import spacy
from spacy.tokenizer import Tokenizer
from pyexcel_ods3 import save_data
from collections import OrderedDict
import io

vocabulary = {}
spam_vocabulary = {}
ham_vocabulary = {}

sentences = {}
spam_senteces = {}
ham_sentences = {}

ham_percentages = {}
spam_percentages = {}



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
        
class EmailInfo():
    def __init__(self, spam, email, id):
        self.info = {
            "id": id,
            "spam": spam,
            "raw": email.text_body,
            "n_cc": len(email.ccs),
            "n_receivers": len(email.receivers)
        }
    
    def set_attribute(self, name, value):
        self.info[name] = value

    def get_info(self):
        return self.info

    def get(self, attribute):
        return self.info[attribute]

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
        self.emails.append(EmailInfo(spam, email, self.length))
        self.length += 1

    def find_email_by_id(self, id):
        for email in self.emails:
            if email.id == id:
                return email

    def get_amount_of_class(self, spam):
        count = 0
        for email in self.emails:
            if email.get("spam") == spam:
                count += 1
        return count

    def delete_email_by_id(self, id):
        self.emails.pop(self.find_email_by_id(id))

    def find_similarities(self):

        for email_1 in self.emails:
            for email_2 in self.emails:
                pass
    
    def similarities_to_class(self, id, spam):
        if spam == 0:
            email = self.find_email_by_id(id)
            pass
        else:
            pass

    def save_data_to_ods(self, name):
        ios = io.StringIO("/home/baserad/Documents/" + name)
        save_info = [email.get_info().values() for email in self.get_emails()]
        save = OrderedDict()

        save.update({"Sheet_1": save_info})
    
        save_data("AttributeData.ods", save)



def load_email(path, file_name):
    with open(os.path.join(path, file_name), "rb") as f:
        return(SimplifiedMessage(e_parser.BytesParser(policy=email.policy.default).parse(f)))

def add_count_to_dict(vocab, word, amount = 1):
    if word in vocab:
        vocab[word] += amount
    else:
        vocab[word] = amount

def word_given_vocab(word, spam):
    if spam:
        return spam_percentages[word]
    else:
        return ham_percentages[word]

def get_normalized_percentages(vocabs, sizes, vocabulary):

    vocab_len = len(vocabs)
    dict_list = [{} for index in range(vocab_len)]
    tot_size = np.sum(sizes)

    for word, count in vocabulary.items():
        new_sizes = []


        for i in range(vocab_len):
            percentage_o_size = sizes[i] / tot_size
            percentage_o_size_inversed = 1 - percentage_o_size
            
            w_count = vocabs[i][word]

            new_count = w_count * percentage_o_size_inversed
            new_sizes.append(new_count)

        new_size_total = np.sum(new_sizes)
        for i in range(len(new_sizes)):
            new_percentage = new_sizes[i] / new_size_total
            dict_list[i][word] = new_percentage
        
    return dict_list

            
if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")
    nlp = spacy.load('en_core_web_lg')
    spam_folder = os.path.join(data_path,"spam")
    ham_folder = os.path.join(data_path, "ham")

    ham_filenames = [name for name in sorted(os.listdir(ham_folder)) if len(name) > 20]
    spam_filenames = [name for name in sorted(os.listdir(spam_folder)) if len(name) > 20]

    ham_emails = [load_email(ham_folder, name) for name in ham_filenames] 
    spam_emails = [load_email(spam_folder, name) for name in spam_filenames]

    tot_emails = {1: spam_emails, 0: ham_emails}

    email_attributes = EmailAttributeList()
    email_attributes.add_emails(0, ham_emails[:50])
    email_attributes.add_emails(1, spam_emails[:50])

    suz_tokenizer = re.compile('\!|\$|(password)|(virus)')

    for email_1 in email_attributes.get_emails():
        print("doing it")
        doc1 = nlp(email_1.get("raw"))
        assert doc1.has_annotation("SENT_START")
        email_1.set_attribute("n_ents", len(doc1.ents))
        email_1.set_attribute("n_tokens", len(doc1))
        
        unique_tokens = []
        capital_words = 0

        
        for token in doc1:
            if token.text not in unique_tokens:
                unique_tokens.append(token.text)

            if token.text.upper() == token.text:
                capital_words += 1

        email_1.set_attribute("capital_words", capital_words)
        email_1.set_attribute("unique_n_tokens", len(unique_tokens))
        email_1.set_attribute("alarming_tokens", len([suz_tokenizer.findall(email_1.get("raw"))]))

        avg_spam_sentence_structure_similarity = 0
        avg_ham_sentence_structure_similarity = 0
        avg_spam_doc_similarity = 0
        avg_ham_doc_similarity = 0

        for email_2 in email_attributes.get_emails():
            doc2 = nlp(email_2.get("raw"))
            """ 
           assert doc1.has_annotation("SENT_START")

            sent_list_1 = list(doc1.sents)
            sent_list_2 = list(doc2.sents)
            
            if len(sent_list_1) < len(sent_list_2):
                for i in range(len(sent_list_1)):
                    if email_2.get("spam") == 0:
                        avg_ham_sentence_structure_similarity += (sent_list_1[i].similarity(sent_list_2[i]) / len(sent_list_1))
                    else:
                        avg_spam_sentence_structure_similarity += (sent_list_1[i].similarity(sent_list_2[i]) / len(sent_list_1))
            else:
                for i in range(len(sent_list_2)):
                    if email_2.get("spam") == 0:
                        avg_ham_sentence_structure_similarity += (sent_list_1[i].similarity(sent_list_2[i]) / len(sent_list_2))
                    else:
                        avg_spam_sentence_structure_similarity += (sent_list_1[i].similarity(sent_list_2[i]) / len(sent_list_2))
            """
            if email_2.get("spam") == 0:
                avg_ham_doc_similarity += doc1.similarity(doc2)
            else:
                avg_spam_doc_similarity += doc1.similarity(doc2)

        avg_ham_sentence_structure_similarity /= email_attributes.get_amount_of_class(0)
        avg_spam_sentence_structure_similarity /= email_attributes.get_amount_of_class(1)
        avg_ham_doc_similarity /= email_attributes.get_amount_of_class(0)
        avg_spam_doc_similarity /= email_attributes.get_amount_of_class(1)
        

        email_1.set_attribute("avg_ham_sentence_structure_similarity", avg_ham_sentence_structure_similarity)
        email_1.set_attribute("avg_spam_sentence_structure_similarity", avg_spam_sentence_structure_similarity)

        email_1.set_attribute("avg_ham_doc_similarity", avg_ham_doc_similarity)
        email_1.set_attribute("avg_spam_doc_similarity", avg_spam_doc_similarity)

    email_attributes.save_data_to_ods("OOAGASASDA")

    


for key, emails in tot_emails.items():
    for email in emails:
        content = ""
        content_list = []
        payload = email.get_payload()

        if email.is_multipart():    
            for part in payload:
                c_type = part.get_content_type()

                if c_type == "text/plain":
                    content_list.extend(part.get_payload())
        else:
            content_list.extend(payload)
        
        content = ''.join(map(str, content_list))

        reg_words = re.compile('[A-z]\w+')
        
        words = reg_words.findall(content)
    
        for word in words:
            word_lower = word.lower()
            add_count_to_dict(vocabulary, word_lower)
            if key == "ham":
                add_count_to_dict(ham_vocabulary, word_lower)
            if key == "spam":
                add_count_to_dict(spam_vocabulary, word_lower)

for word in vocabulary.keys():
    add_count_to_dict(ham_vocabulary, word, amount=0.00000001)
    add_count_to_dict(spam_vocabulary, word, amount=0.00000001)

ham_percentages, spam_percentages = get_normalized_percentages([ham_vocabulary, spam_vocabulary], [len(ham_emails), len(spam_emails)], vocabulary)
print(ham_percentages)
print(word_given_vocab("lol", spam=False))
print(word_given_vocab("lol", spam=True))

soreted_ham = sorted(ham_percentages, key = ham_percentages.get, reverse=True)
removed_values = []
for key in soreted_ham:
    if word_given_vocab(key, spam=True) < 0.3:
        removed_values.append(key)

for value in removed_values:
    soreted_ham.remove(value)


i = 0
for key in soreted_ham:
    i += 1
    print()
    print(key)
    print(f"for ham: {word_given_vocab(key, spam=False)}") 
    print(f"for spam: {word_given_vocab(key, spam=True)}") 
    print()
    if i > 20:
        break



