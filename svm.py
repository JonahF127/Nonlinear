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


# function to count total characters in the email
def character_count(email):
    return len(email)


# function to count number of letter characters in the email
def letter_character_count(email):
    # remove punctuation
    text = email.translate(str.maketrans('', '', string.punctuation))

    # remove spaces and newlines/tabs
    text = text.replace(" ", "").replace("\n", "").replace("\t", "")

    # keep only alphabetic characters
    letters_only = ""
    for ch in text:
        if ch.isalpha():
            letters_only += ch

    # return number of letters
    return len(letters_only)


# function to count number of flagged words in an email
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
        "won",
        "money",
        "offer",
        "prize",
        "click",
        "text",
        "txt",
        "call",
        "subscription",
        "subscribe",
        "sex",
        ]
    
    # Compare the email with flagged_words
    words = lowercase_text.split()
    flagged_word_count = sum(1 for word in words if word in flagged_words)
    return flagged_word_count


# function to count number of flagged bigrams in an email
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
        "have won"
    ]


    # Compare the email with flagged_bigrams 
    words = lowercase_text.split()
    bigrams = []
    for i in range(len(words) - 1):
        bigrams.append(f"{words[i]} {words[i+1]}")
    
    flagged_bigram_count = sum(1 for bigram in bigrams if bigram in flagged_bigrams)
    return flagged_bigram_count

    

    
# function to count number of digits in an email
def count_digits(email):
    # remove all punctuation from the email
    text = email.translate(str.maketrans('', '', string.punctuation))
    

    # remove all white spaces
    no_whitespace = "".join(text.split())

    # count number of digits
    digit_count = 0
    for ch in no_whitespace:
        if ch.isdigit():
            digit_count += 1

    return digit_count


# function to count number of (specified) special characters in an email
def count_specials(email):
    # remove all white spaces
    no_whitespace = "".join(text.split())

    # Make list of flagged special characters
    flagged_specials = [
        "/",
        "@",
    ]
    
    # Count the number of flagged specials
    special_count = 0
    for ch in no_whitespace:
        if not(ch.isalpha()) and not(ch.isdigit()):
                for special in flagged_specials:
                    if ch == special:
                        special_count += 1
            

    return special_count



# check how many times email includes a link
def count_links(email):
    # remove all punctuation from the email
    text = email.translate(str.maketrans('', '', string.punctuation))

    # get list of words
    words = text.split()

    # find number of links
    links = 0
    link_ends = [
        "www.",
        ".com",
        ".co",
        ".uk",
        ".net"
    ]
    
    for word in words:
        if "www" in word:
            links += 1

    return links


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

    # create column for number of characters
    emails["character_count"] = emails["Message"].apply(character_count)

    # create column for number of letters in email
    emails["letter_count"] = emails["Message"].apply(letter_character_count)

    # create column for number of flagged words in email
    emails["flagged_word_count"] = emails["Message"].apply(flagged_words_count)

    # create column for number of flagged bigrams in email
    emails["flagged_bigrams_count"] = emails["Message"].apply(flagged_bigrams_count)

    # create column for counting number of digits in email
    emails["digit_count"] = emails["Message"].apply(count_digits)

    # create column for counting number of links in email
    emails["links_count"] = emails["Message"].apply(count_links)


    # save dataframe to new csv
    emails.to_csv('email_data.csv', index=False)

    




if __name__ == "__main__":
    main()