import pandas
import string

# function to count number of words in an email
def find_num_words(email):
    # remove all punctuation from the email
    text = email.translate(str.maketrans('', '', string.punctuation))

    # return number of words in text
    return len(text.split())


# function to count number of fully uppercase words in the email
def uppercase_count(email):
    # remove punctuation from email
    text = email.translate(str.maketrans('', '', string.punctuation))

    # create list of words
    words = text.split()

    # count number of upper case words
    upper_count = 0
    for word in words:
        if word.isupper():
            upper_count += 1
    
    return upper_count


# function to count number of unique words in an email
def unique_word_count(email):
    # remove punctuation from email
    text = email.translate(str.maketrans('', '', string.punctuation))
    
    # make text lowercase
    lowercase_text = text.lower()

    # create set of words, which will remove duplicates
    words = set(lowercase_text.split())

    # return number of unique words
    return len(words)
    







def main():
    # read in the csv as pandas dataframe
    # the columns are "Category" and "Message"
    emails = pandas.read_csv("email.csv")

    # create column for the number of words in each email
    emails["word_count"] = emails["Message"].apply(find_num_words)

    # create column for number of uppercase words in each email
    emails["uppercase_word_count"] = emails["Message"].apply(uppercase_count)
 
    # create column for number of unique words in each email
    emails["unique_word_count"] = emails["Message"].apply(unique_word_count)


    # save dataframe to new csv
    emails.to_csv('email_data.csv', index=False)

    




if __name__ == "__main__":
    main()