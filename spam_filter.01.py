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
from spacytextblob.spacytextblob import SpacyTextBlob

from spam_filter_classes import SimplifiedMessage, EmailInfo, EmailAttributeList

vocabulary = {}
spam_vocabulary = {}
ham_vocabulary = {}


def load_email(path, file_name):
    with open(os.path.join(path, file_name), "rb") as f:
        return(SimplifiedMessage(e_parser.BytesParser(policy=email.policy.default).parse(f)))

def add_to_vocabulary(vocab, word, amount = 1):
    if word in vocab:
        vocab[word] += amount
    else:
        vocab[word] = amount

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")
    nlp = spacy.load('en_core_web_lg')
    nlp.add_pipe('spacytextblob')
    spam_folder = os.path.join(data_path,"spam")
    ham_folder = os.path.join(data_path, "ham")

    ham_filenames = [name for name in sorted(os.listdir(ham_folder)) if len(name) > 20]
    spam_filenames = [name for name in sorted(os.listdir(spam_folder)) if len(name) > 20]

    ham_emails = [load_email(ham_folder, name) for name in ham_filenames] 
    spam_emails = [load_email(spam_folder, name) for name in spam_filenames]

    tot_emails = {1: spam_emails, 0: ham_emails}

    email_attributes = EmailAttributeList()
    email_attributes.add_emails(0, ham_emails[:300])
    email_attributes.add_emails(1, spam_emails[:300])
    
    suz_tokenizer = re.compile('\!|\$|(password)|(virus)')

    for email_1 in email_attributes.get_emails():
        doc1 = nlp(email_1.get("raw"))
        assert doc1.has_annotation("SENT_START")


        tag_counts = {
            "NNP": 0,
            "VBZ": 0,
            "VBG": 0,
            "IN": 0,
            "NN": 0,
            "$": 0,
            "CD": 0,
            "Unknown_tags": 0
        }
            
        unique_tokens = []
        capital_words = 0

        total_words = len(doc1)
        if total_words == 0:
            total_words = 1

        for token in doc1:
            add_to_vocabulary(vocabulary, token.text)
            if doc1.get("spam"):
                add_to_vocabulary(spam_vocabulary, token.text)
            else:
                add_to_vocabulary(ham_vocabulary, token.text)
    

            try:
                tag_counts[token.tag] += 1
            except: 
                tag_counts["Unknown_tags"] += 1
            
            if token.text not in unique_tokens:
                unique_tokens.append(token.text)

            if token.text.upper() == token.text:
                capital_words += 1
    

        #more or less syntactic measures
        email_1.set_attribute("n_ents", len(doc1.ents))
        email_1.set_attribute("n_tokens", total_words)
        email_1.set_attribute("n_ents_relative", len(doc1.ents) / total_words)
        email_1.set_attribute("capital_words", capital_words)
        email_1.set_attribute("capital_words_relative", capital_words / total_words)
        email_1.set_attribute("unique_n_tokens", len(unique_tokens))
        email_1.set_attribute("unique_n_tokens_relative", len(unique_tokens) / total_words)
        alarming_tokens_tot = len([suz_tokenizer.findall(email_1.get("raw"))])
        email_1.set_attribute("alarming_tokens", alarming_tokens_tot)
        email_1.set_attribute("alarming_tokens_relative", alarming_tokens_tot / total_words)

        #more semantic measures
        email_1.set_attribute("subjectivity", doc1._.blob.subjectivity)
        email_1.set_attribute("polarity", doc1._.blob.polarity)

        for tag, count in tag_counts.items():
            email_1.set_attribute(tag, count)
            email_1.set_attribute(f"{tag}_relative", count / total_words)
        
    for key in vocabulary:
        add_to_vocabulary(vocabulary, key, 0.002)
        add_to_vocabulary(ham_vocabulary, key, 0.001)
        add_to_vocabulary(spam_vocabulary, key, 0.001)

    email_attributes.save()




