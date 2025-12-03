import streamlit as st
import csv
import numpy as np
import add_features

st.title("Spam Email Classifier")


def get_email():
    email = st.text_input("Enter an email.")

def main(): 
    with open("solution.csv", "r", newline='') as f:
        reader = csv.reader(f)
        next(reader)
        second_line = next(reader)
        solution_list = [float(value) for value in second_line]
        beta = solution_list.pop(len(solution_list) - 1)
        alpha = np.array(solution_list)
        get_email()
        

    

if __name__ == "__main__":
    main()