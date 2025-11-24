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

# function to count total characters in the email (including spaces and punctuation)
def character_count(email):
    return len(email)

def flagged_words_count(email):
    # remove punctuation from email
    text = email.translate(str.maketrans('', '', string.punctuation))
    
    # make text lowercase
    lowercase_text = text.lower()

    # Make the list of common words in spam
    flagged_words = [
        "free",
        "new",
        "now",
        "immediately",
        "urgent",
        "final",
        "premium",
        "win",
        "money",
        "offer",
        "prize",
        "click",
        "text",
        "txt",
        "call",
        "subscription",
        "subscribe"
        ]
    
    # Compare the email with flagged_words (To Do)

def flagged_bigrams_count(email):
    # remove punctuation from email
    text = email.translate(str.maketrans('', '', string.punctuation))
    
    # make text lowercase
    lowercase_text = text.lower()

    # make the list of common bigrams in spam
    flagged_bigrams = [
        "click here",
        "limited time",
        "act now",
        "Have won"
    ]

    # Compare the email with flagged_bigrams (To Do)

    

    







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