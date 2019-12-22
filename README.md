# Parts of Speech Tagging

## Posterior Probability
We calculate the sum of the pos and words and normalize it. We calculate the posterior probability using the emission probability for bayes net. For the bayes net we calculate the probability using the emission probability. For the viterbi algoritm we use emission probabilty and transition probabilty to calculate the posterior probability. For the Gibbs sampling we calculate the probability using the emission, transions probabilty and using the POS count which has been normalzied for the training data. If the word pos is not there then we chose a constant using the minimum possible product. We sum up these probabilities and return for the respective model.

### Bayes Net
We calculated the Bayes net probability and stored it in a dictionary. We stored probability for each word and stored all the possible POS's of the word and normalize them for each of the words. If the testing label exists in the training dictionary, we use the most probable POS from the training data. Else we consider it as a foreign word( 'x' ).

#### Emission Probability
The emission probability of each word and stored it in a dictionary with a tuple of word and pos as key and its probability as the value which again is normalized for each word.

#### Transition Probability
The transition probability is calculated for 12 x 12 POS. It has a tuple of previous_pos and current_pos as keys and its count as values. Later the count is normalized using the total counts.

#### Initial Probability
The initial probability distribution for the first words using their POS. It is again normalized for the first word and used as it is.

### Viterbi
We store the respective combinations of initial distribution and emission probability for the first word. For all the next words we calculate the respective products of emission probability, previous probability and transition probability. While iterating through all the 12 POS we consider the pos with the maximum probability. We append this value to the current distribution. We append such values for all 12 POS up to the length of the sentence. While calculating the probability for the first POS, if the word doesn't exist we append the most frequent first character in the initial probability. While iterating through the second word if we don't find any combination of a word with all of the 12 pos, only then we consider the exception case of using the transition probability, previous probability and a scaling constant because of not having an emission probability.
Backtracking For the last word, we chose the maximum probability for all the 12 POS. After choosing this POS we backtrack from the last word and use the maximum transition probability for that POS. For backtracking, if we don't find a combination of word and pos in the current distribution we consider the transition probability as an alternative. Then we reverse the list because the list was appended in the reverse format and return it.

### MCMC
For Gibbs sampling, we store the emission probabilities for the 12 POS. If the POS doesn't exist we replace it with a downscaled value among all the minimum values. Then we randomly iterate for 1000 times and based on a range of probability it lies in. We do this for all the words in the sentence. Then we store the last of all the POS and return it.

Handling all the exceptions whenever the testing data had some new words or the length of sentence was exceptionally small was another challenege.This program in particularly required a lot of dictioneries to be maintained.

### Run the code

Run the code using following command on cmd, 

./label.py bc.test bc.train
