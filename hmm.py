import numpy as np


class HiddenMarkovModel:
    """
    A basic implementation of a Hidden Markov Model (HMM)
    with discrete states and discrete observations.
    """

    def __init__(self, states, observations, transition_prob, emission_prob, init_prob):
        """
        Parameters
        ----------
        states : array-like
            List of possible hidden states.

        observations : array-like
            List of possible observation symbols.

        transition_prob : 2D array
            Matrix where entry (i, j) = P(s_t = j | s_{t-1} = i).

        emission_prob : 2D array
            Matrix where entry (i, k) = P(obs = k | state = i).

        init_prob : 1D array
            Initial distribution over hidden states.
        """
        self.states = np.array(states)
        self.observations = np.array(observations)

        self.S = len(states)            # number of states
        self.O = len(observations)      # number of observation symbols

        self.A = np.array(transition_prob)
        self.B = np.array(emission_prob)
        self.pi = np.array(init_prob)

    # ----------------------------------------------------------------------
    # Forward Algorithm
    # ----------------------------------------------------------------------
    def likelihood_forward(self, obs_seq):
        """
        Compute the likelihood of an observation sequence using the
        Forward algorithm.

        Returns
        -------
        prob : float
            Total probability of the sequence.

        alpha : ndarray [N x T]
            Forward messages.
        """
        T = len(obs_seq)
        alpha = np.zeros((self.S, T))

        # Convert symbols to indices
        obs_idx = [self._obs_index(o) for o in obs_seq]

        # Initial step
        alpha[:, 0] = self.pi * self.B[:, obs_idx[0]]

        # Recursive updates
        for t in range(1, T):
            alpha[:, t] = (alpha[:, t - 1] @ self.A) * self.B[:, obs_idx[t]]

        # Sequence likelihood
        return alpha[:, -1].sum(), alpha

    # ----------------------------------------------------------------------
    # Backward Algorithm
    # ----------------------------------------------------------------------
    def likelihood_backward(self, obs_seq):
        """
        Compute the likelihood of an observation sequence using the
        Backward algorithm.

        Returns
        -------
        prob : float
            Total probability of the sequence.

        beta : ndarray [N x T]
            Backward messages.
        """
        T = len(obs_seq)
        beta = np.zeros((self.S, T))

        obs_idx = [self._obs_index(o) for o in obs_seq]

        # Initialize
        beta[:, -1] = 1

        # Backward recursion
        for t in range(T - 2, -1, -1):
            beta[:, t] = self.A @ (self.B[:, obs_idx[t + 1]] * beta[:, t + 1])

        # Final probability
        prob = np.sum(self.pi * self.B[:, obs_idx[0]] * beta[:, 0])
        return prob, beta

    # ----------------------------------------------------------------------
    # Sequence Likelihood
    # ----------------------------------------------------------------------
    def likelihood(self, obs_seq):
        """Returns the likelihood of an observation sequence."""
        prob, _ = self.likelihood_forward(obs_seq)
        return prob

    # ----------------------------------------------------------------------
    # Viterbi Algorithm
    # ----------------------------------------------------------------------
    def decode(self, obs_seq):
        """
        Viterbi decoding to obtain the most likely hidden state sequence.

        Returns
        -------
        path : ndarray of states
        prob : float, max path probability
        """
        T = len(obs_seq)
        obs_idx = [self._obs_index(o) for o in obs_seq]

        delta = np.zeros((self.S, T))
        psi = np.zeros((self.S, T), dtype=int)

        # Initialization
        delta[:, 0] = self.pi * self.B[:, obs_idx[0]]

        # Recursion
        for t in range(1, T):
            scores = delta[:, t - 1].reshape(-1, 1) * self.A
            psi[:, t] = np.argmax(scores, axis=0)
            delta[:, t] = scores.max(axis=0) * self.B[:, obs_idx[t]]

        # Backtracking
        states_idx = np.zeros(T, dtype=int)
        states_idx[-1] = np.argmax(delta[:, -1])

        for t in range(T - 2, -1, -1):
            states_idx[t] = psi[states_idx[t + 1], t + 1]

        return self.states[states_idx], delta[:, -1].max()

    # ----------------------------------------------------------------------
    # Baum–Welch (EM)
    # ----------------------------------------------------------------------
    def learn(self, obs_seq, iterations=1):
        """
        Train the HMM using the Baum–Welch EM algorithm.

        Parameters
        ----------
        iterations : int
            Number of EM updates.
        """
        obs_idx = np.array([self._obs_index(o) for o in obs_seq])
        T = len(obs_seq)

        for _ in range(iterations):

            prob, alpha = self.likelihood_forward(obs_seq)
            _, beta = self.likelihood_backward(obs_seq)

            # State posterior (gamma)
            gamma = alpha * beta
            gamma /= gamma.sum(axis=0)

            # Transition posterior (xi)
            xi = np.zeros((self.S, self.S, T - 1))
            for t in range(T - 1):
                temp = (
                    alpha[:, t][:, None]
                    * self.A
                    * self.B[:, obs_idx[t + 1]][None, :]
                    * beta[:, t + 1][None, :]
                )
                xi[:, :, t] = temp / temp.sum()

            # M-step: update parameters
            self.pi = gamma[:, 0]

            self.A = xi.sum(axis=2) / gamma[:, :-1].sum(axis=1, keepdims=True)

            for k in range(self.O):
                mask = (obs_idx == k)
                self.B[:, k] = (gamma[:, mask].sum(axis=1) /
                                gamma.sum(axis=1))

    # ----------------------------------------------------------------------
    # Utility
    # ----------------------------------------------------------------------
    def _obs_index(self, obs):
        """Get observation index in the observation list."""
        return np.where(self.observations == obs)[0][0]