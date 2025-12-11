# Hidden-Pollution-With-Hidden-Markov-Model
Air pollution measurements are influenced by weather conditions such as rainfall,
but rain is not recorded in the UCI Air Quality dataset [ 6]. In this project we use a
two-state hidden Markov model (HMM) to infer whether each hourly observation
corresponds to a rainy or not rainy regime based solely on pollutant concentrations
(CO, NO2, O3). Rather than learning parameters, we construct a fixed HMM
in which emission probabilities are derived from AQI categories and transition
probabilities are set a priori. We generate synthetic ground-truth labels using
Forward Filtering Backward Sampling (FFBS) and then apply Viterbi decoding
to estimate the most likely sequence of latent regimes. Experiments with the
dataset-derived emission probabilities show that the model collapses to a trivial
solution and predicts only the not rainy state. When emission probabilities for
low-AQI categories are increased, the HMM produces meaningful latent-state
dynamics and achieves substantially higher precision, recall, and F1. These results
highlight the sensitivity of HMM inference to emission design and illustrate both
the strengths and limitations of using simple discrete HMMs to infer unobserved
weather conditions from air quality observations.
### CSE 150A Final Project



Benjamin Ng, Miguel Santos, Kanishk Hari, Derek Huang
