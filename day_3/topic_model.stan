data {
  int<lower=1> D;          // Number of documents
  int<lower=1> V;          // Vocabulary size
  int<lower=1> N;          // Total number of word occurrences
  array[N] int<lower=1, upper=D> doc;  // Document indices for each word
  array[N] int<lower=1, upper=V> word; // Word indices for each word
  int<lower=1> K;          // Number of topics
}

parameters {
  array[D] simplex[K] theta; // Topic proportions per document
  array[K] simplex[V] beta;  // Word probabilities per topic
}

model {
  // Priors
  for (k in 1:K)
    beta[k] ~ dirichlet(rep_vector(0.1, V)); // Topic-word distribution prior
  for (d in 1:D)
    theta[d] ~ dirichlet(rep_vector(0.1, K)); // Document-topic distribution prior

  // Likelihood
  for (n in 1:N) {
    real word_prob = 0;
    for (k in 1:K)
      word_prob += theta[doc[n]][k] * beta[k][word[n]];
    target += log(word_prob);
  }
}
