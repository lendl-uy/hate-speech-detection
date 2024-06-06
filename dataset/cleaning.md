# Cleaning Dataset Versions

## Version 0 <i>(contains duplicate tweets)</i>
Preprocessing steps:
- Removed newline characters
- Removed URLs
- Removed usernames
- Removed completely numeric words
- Removed HTML tags
- Put all texts in lowercase
- Remove "rt" from tweets
- Shorten all variations of "haha"
- Removed ALL hashtags (including text)
- Expanded contractions
- Removed possessives
- Removed punctuations*
- Manually replaced selected words (mainly standardized the expletives)
- Removed English and Filipino stopwords (provided by stopwordsiso and modified based on observed word frequencies and context of our problem)
  - added stopwords: 'si','kay','lang','yung','wag','ba','yan','iyan','kayo','pag','naman','mo','niyo','nung','kang','tong','nalang'
  - removed stopwords: 'not','di','hindi','wala'


Notes:
- Emoticons like :) and :( (along with its variations) were considered to be kept but they were ultimately removed since they only comprised 1% of the total corpus.
- Past studies either manually segmented the hashtags or completely removed them. Cabasag et al. (2019) showed no significant decline in model performance when removing the hashtags so they are removed in this study


## Version 1 <i>(no duplicates)</i>
All processing steps done in Version 0 + removal of duplicate tweets
If the same tweet appears in the train, val, and test sets, copies of the tweets are first removed from the train set and then the validation set.

## Version 2 <i>(no candidate names)</i>
Removed the candidate names to allow us to measure the performance of our models beyond the context of the 2016 PH elections.

Removed terms: "jejomar", "binay", "mar", "roxas", "rodrigo", "duterte", "grace", "poe", "miriam", "defensor", "santiago" 
