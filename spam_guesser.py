import streamlit as st
import pandas as pd
import csv
import numpy as np
import add_features
import prepare_data

st.title("Spam Email Classifier")

def create_vector(email):
    data = [
        add_features.find_num_words(email),
        add_features.uppercase_count(email),
        add_features.unique_word_count(email),
        add_features.character_count(email),
        add_features.letter_character_count(email),
        add_features.flagged_words_count(email),
        add_features.flagged_bigrams_count(email),
        add_features.count_digits(email),
        add_features.count_links(email),
        add_features.free_count(email),
        add_features.count_specials(email)
    ]

    email_vec = np.array(data)
    return email_vec


def get_email():
    email = st.text_input("Enter an email.")
    return email


def make_prediction(alpha, beta, email_vec):
    result = np.dot(alpha, email_vec)
    if result >= beta:
        return 1
    else:
        return -1
  

def main(): 
    df = pd.read_csv("email_data.csv")
    feature_cols = [
        "word_count",
        "uppercase_word_count",
        "unique_word_count",
        "character_count",
        "letter_count",
        "flagged_word_count",
        "flagged_bigrams_count",
        "digit_count",
        "links_count",
        "free_count",
        "special_count"
    ]
    X = df[feature_cols].to_numpy(dtype=float)
    min_vals = np.min(X, axis = 0)
    max_vals = np.max(X, axis = 0)
    
    with open("solution.csv", "r", newline='') as f:
        reader = csv.reader(f)
        next(reader)
        second_line = next(reader)
        solution_list = [float(value) for value in second_line]
    beta = solution_list.pop(len(solution_list) - 1)
    alpha = np.array(solution_list)
        
    email = get_email()
    if st.button("Submit Email"):
        if email:
            email_vec = create_vector(email)
            scaled_vec = (email_vec - min_vals) / (max_vals - min_vals)
            
            
            prediction = make_prediction(alpha, beta, scaled_vec)
            if prediction == 1:
                st.write("The email is not spam!")
            else:
                st.write("The email is spam")
            
                
        else:
            st.warning("Please enter a valid email")
        

    

if __name__ == "__main__":
    main()