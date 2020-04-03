import numpy as np


def indexize_observation(seq_str):
    mapping = {
        'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5
    }
    return [mapping[c] for c in seq_str]


def get_samples(path):
    return [indexize_observation(line.strip()) for line in open(path, 'r').readlines()]


class HMMModel:

    def __init__(self, n_states, n_observation,
                 startprob=None, transmat=None, emissionprob=None):
        self.n_states = n_states
        self.n_observation = n_observation
        self.startprob = startprob
        self.transmat = transmat
        self.emissionprob = emissionprob

    def load(self, file):
        initial_head = 'initial: '
        transition_head = 'transition: '
        observation_head = 'observation: '
        phase = None
        tmp = []
        counter = 0
        for line in open(file, 'r').readlines():
            line = line.strip()
            if not line:
                continue

            if line.startswith(initial_head):
                n_states = int(line.replace(initial_head, ''))
                phase = 'INIT'
            elif line.startswith(transition_head):
                n_states = int(line.replace(transition_head, ''))
                phase = 'TRANS'
            elif line.startswith(observation_head):
                n_states = int(line.replace(observation_head, ''))
                phase = 'OBS'
            else:
                tmp.append(np.array([float(s) for s in line.split('\t')]))
                counter += 1
                if phase == 'INIT' and counter == 1:
                    self.startprob = tmp[0]
                elif phase == 'TRANS' and counter == self.n_states:
                    self.transmat = np.array(tmp)
                elif phase == 'OBS' and counter == self.n_observation:
                    self.emissionprob = np.array(tmp)
                else:
                    continue
                tmp = []
                phase = None
                counter = 0

            if self.n_states != n_states:
                raise Exception

    def save(self, path):
        with open(path, 'w') as fw:
            fw.write('initial: {}\n'.format(self.n_states))
            fw.write('\t'.join([str(e) for e in self.startprob.tolist()]) + '\n')
            fw.write('\n')
            fw.write('transition: {}\n'.format(self.n_states))
            for l in self.transmat.tolist():
                fw.write('\t'.join([str(e) for e in l]) + '\n')
            fw.write('\n')
            fw.write('observation: {}\n'.format(self.n_states))
            for l in self.emissionprob.tolist():
                fw.write('\t'.join([str(e) for e in l]) + '\n')

    def get_likelihood(self, observ_seq):
        delta_list = []
        # init
        init_observ = observ_seq[0]
        init_delta_row = self.emissionprob[init_observ, :] * self.startprob
        delta_list.append(init_delta_row)
        # iterate
        for observ in observ_seq[1:]:
            transition_part = self.transmat * np.reshape(delta_list[-1], (-1, 1))
            max_values = np.max(transition_part, axis=0)
            delta_row = self.emissionprob[observ, :] * max_values
            delta_list.append(delta_row)
        # terminate
        return np.max(delta_list[-1])

    def _get_update_parameters(self, observ_seq):
        # Forward Procedure
        alpha_list = []
        init_observ = observ_seq[0]
        init_alpha_row = self.emissionprob[init_observ, :] * self.startprob
        alpha_list.append(init_alpha_row)
        for observ in observ_seq[1:]:
            alpha_row = alpha_list[-1].dot(self.transmat) * self.emissionprob[observ, :]
            alpha_list.append(alpha_row)
        alpha_matrix = np.array(alpha_list).T

        # Backward Procedure
        beta_list = []
        init_beta_row = np.ones(self.n_states)
        beta_list.insert(0, init_beta_row)
        for next_observ in observ_seq[:0:-1]:
            beta_row = self.transmat.dot(self.emissionprob[next_observ, :] * beta_list[0])
            beta_list.insert(0, beta_row)
        beta_matrix = np.array(beta_list).T

        # calculate gamma
        alpha_times_beta = alpha_matrix * beta_matrix
        gamma_matrix = alpha_times_beta / np.sum(alpha_times_beta, axis=0)

        # calculate epsilon
        epsilon_matrix_list = []
        for t in range(len(alpha_list)-1):
            alpha_t = alpha_list[t].reshape((-1, 1))
            observ_next_t = observ_seq[t+1]
            b_next_t = self.emissionprob[observ_next_t, :].reshape((1,-1))
            beta_next_t = beta_list[t+1].reshape((1,-1))
            matrix = alpha_t * self.transmat * b_next_t * beta_next_t
            epsilon_matrix_list.append(
                matrix / np.sum(matrix)
            )

        # get update parameters
        init_gamma = gamma_matrix[:, 0]
        num_of_transition_from_i_to_j = np.sum(epsilon_matrix_list, axis=0)
        number_of_visiting_state_i = np.sum(gamma_matrix, axis=1)
        number_of_observation_k_in_state_i_at_k = []
        for k in range(self.n_observation):
            mask = np.where(np.array(observ_seq) == k, 1, 0)
            number_of_observation_k_in_state_i_at_k.append(
                np.sum(gamma_matrix * mask, axis=1)
            )
        number_of_observation_k_in_state_i = np.array(number_of_observation_k_in_state_i_at_k)
        return (init_gamma, num_of_transition_from_i_to_j,
                number_of_visiting_state_i, number_of_observation_k_in_state_i)

    def _update(self, samples):
        param_collect = {
            'init_gamma': [],
            'num_of_transition_from_i_to_j': [],
            'number_of_visiting_state_i': [],
            'number_of_observation_k_in_state_i': [],
        }
        for observ_seq in samples:
            init_gamma, num_of_transition_from_i_to_j, \
                number_of_visiting_state_i, number_of_observation_k_in_state_i = \
                    self._get_update_parameters(observ_seq)
            param_collect['init_gamma'].append(init_gamma)
            param_collect['num_of_transition_from_i_to_j'].append(num_of_transition_from_i_to_j)
            param_collect['number_of_visiting_state_i'].append(number_of_visiting_state_i)
            param_collect['number_of_observation_k_in_state_i'] \
                .append(number_of_observation_k_in_state_i)
        self.startprob = np.mean(param_collect['init_gamma'], axis=0)
        self.transmat = np.sum(param_collect['num_of_transition_from_i_to_j'], axis=0) \
            / np.sum(param_collect['number_of_visiting_state_i'], axis=0).reshape((-1, 1))
        self.emissionprob = np.sum(param_collect['number_of_observation_k_in_state_i'], axis=0) \
            / np.sum(param_collect['number_of_visiting_state_i'], axis=0).reshape((-1,))

    def fit(self, samples, n_iter=20):
        for i in range(n_iter):
            print(i)
            self._update(samples)


def test_get_likelihood():
    startprob = np.array([1.0, 0.0, 0.0])
    transmat = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    emissionprob = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    hmm = HMMModel(n_states=3, n_observation=2,
                   startprob=startprob, transmat=transmat, emissionprob=emissionprob)

    observ_seq_1 = np.array([0, 0, 1, 1, 1])
    expected_1 = np.array([0, 0, 1])
    result_1 = hmm.get_likelihood(observ_seq_1)
    np.testing.assert_array_equal(result_1, expected_1)
    observ_seq_2 = np.array([0, 0])
    expected_2 = np.array([0, 1, 0])
    result_2 = hmm.get_likelihood(observ_seq_2)
    np.testing.assert_array_equal(result_2, expected_2)


def main():
    n_states = 6
    n_observation = 6
    acc_list = []

    train_seq_paths = ['train_seq_01.txt', 'train_seq_02.txt', 'train_seq_03.txt', 
                       'train_seq_04.txt', 'train_seq_05.txt']
    samples_li = [
        get_samples('data/' + path) for path in train_seq_paths
    ]
    model_li = [
        line.strip() for line in open('modellist.txt', 'r').readlines()
    ]

    hmm_li = [
        HMMModel(n_states=n_states, n_observation=n_observation)
        for _ in range(len(train_seq_paths))
    ]
    for hmm in hmm_li:
        hmm.load('model_init.txt')

    test_samples = get_samples('data/test_seq.txt')

    for k in range(500):
        print('=={}=='.format(k))

        for i, hmm in enumerate(hmm_li):
            samples = samples_li[i]
            hmm.fit(samples, n_iter=1)
            hmm.save('result/' + model_li[i])

        result = ''
        for observ in test_samples:
            likehoods = []
            for hmm in hmm_li:
                likehoods.append(hmm.get_likelihood(observ))
            ind = np.argmax(likehoods)
            result += '{}\n'.format(model_li[ind])

        with open('result/predict_lbl.txt', 'w') as fw:
            fw.write(result)

        ans = open('data/test_lbl.txt', 'r').read()
        acc = np.mean([1 if a == b else 0 for a, b in zip(ans.split('\n'), result.split('\n'))])
        acc_list.append(acc)

        with open('result/acc.txt', 'w') as fw:
            for acc in acc_list:
                fw.write(str(acc) + '\n')


def test():
    test_get_likelihood()


if __name__ == '__main__':
    main()
