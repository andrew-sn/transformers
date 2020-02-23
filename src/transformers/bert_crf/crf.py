__version__ = '0.7.2'

from typing import List, Optional

import torch
import torch.nn as nn
import numpy as np
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class CRF(nn.Module):
    """Conditional random field.
    This module implements a conditional random field [LMP01]_. The forward computation
    of this class computes the log likelihood of the given sequence of tags and
    emission score tensor. This class also has `~CRF.decode` method which finds
    the best tag sequence given an emission score tensor using `Viterbi algorithm`_.
    Args:
        num_tags: Number of tags.
        batch_first: Whether the first dimension corresponds to the size of a minibatch.
    Attributes:
        start_transitions (`~torch.nn.Parameter`): Start transition score tensor of size
            ``(num_tags,)``.
        end_transitions (`~torch.nn.Parameter`): End transition score tensor of size
            ``(num_tags,)``.
        transitions (`~torch.nn.Parameter`): Transition score tensor of size
            ``(num_tags, num_tags)``.
    .. [LMP01] Lafferty, J., McCallum, A., Pereira, F. (2001).
       "Conditional random fields: Probabilistic models for segmenting and
       labeling sequence data". *Proc. 18th International Conf. on Machine
       Learning*. Morgan Kaufmann. pp. 282–289.
    .. _Viterbi algorithm: https://en.wikipedia.org/wiki/Viterbi_algorithm
    """

    def __init__(self, num_tags: int, batch_first: bool = False, init_transitions=False) -> None:
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.rand(num_tags))  # rand --> empty
        self.end_transitions = nn.Parameter(torch.rand(num_tags))
        if init_transitions:
            logging.info('注意: 初始化转移矩阵目前只适用于BIO体系...')
            self.transitions = self._build_transition_matrix((num_tags-1)//2)
        else:
            self.transitions = nn.Parameter(torch.rand(num_tags, num_tags))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the transition parameters.
        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.
        """
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def forward(
            self,
            emissions: torch.Tensor,
            tags: torch.LongTensor,
            mask: Optional[torch.ByteTensor] = None,
            reduction: str = 'sum',
    ) -> torch.Tensor:
        """Compute the conditional log likelihood of a sequence of tags given emission scores.
        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            tags (`~torch.LongTensor`): Sequence of tags tensor of size
                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            reduction: Specifies  the reduction to apply to the output:
                ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
                ``sum``: the output will be summed over batches. ``mean``: the output will be
                averaged over batches. ``token_mean``: the output will be averaged over tokens.
        Returns:
            `~torch.Tensor`: The log likelihood. This will have size ``(batch_size,)`` if
            reduction is ``none``, ``()`` otherwise.
        """
        self._validate(emissions, tags=tags, mask=mask)
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, mask)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask)
        # shape: (batch_size,)
        llh = numerator - denominator

        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        assert reduction == 'token_mean'
        return llh.sum() / mask.float().sum()

    def decode(self, emissions: torch.Tensor,
               mask: Optional[torch.ByteTensor] = None,
               topk=1) -> List[List[int]]:
        """Find the most likely tag sequence using Viterbi algorithm.
        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            topk (`int`): multi-decode, default=1, return the path with highest possibility.
        Returns:
            List of list containing the best tag sequence for each batch.
        """
        self._validate(emissions, mask=mask)
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        if topk > 1:
            return self._viterbi_multi_decode(emissions, mask, topk)
        return self._viterbi_decode(emissions, mask)

    def _validate(
            self,
            emissions: torch.Tensor,
            tags: Optional[torch.LongTensor] = None,
            mask: Optional[torch.ByteTensor] = None) -> None:
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f'expected last dimension of emissions is {self.num_tags}, '
                f'got {emissions.size(2)}')

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    'the first two dimensions of emissions and tags must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}')

        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    'the first two dimensions of emissions and mask must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}')
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')

    def _compute_score(
            self, emissions: torch.Tensor, tags: torch.LongTensor,
            mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.shape[:2] == tags.shape
        assert emissions.size(2) == self.num_tags
        assert mask.shape == tags.shape
        assert mask[0].all()

        seq_length, batch_size = tags.shape
        mask = mask.float()

        # Start transition score and first emission
        # shape: (batch_size,)
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]

            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        # End transition score
        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        # shape: (batch_size,)
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        # shape: (batch_size,)
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(
            self, emissions: torch.Tensor, mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length = emissions.size(0)

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions: torch.FloatTensor,
                        mask: torch.ByteTensor) -> List[List[int]]:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        history = []

        # score is a tensor of size (batch_size, num_tags) where for every batch,
        # value at column j stores the score of the best tag sequence so far that ends
        # with tag j
        # history saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            # [B, T, 1] + [T, T] + [B, 1, T] --> [B, T, T]
            next_score = broadcast_score + self.transitions + broadcast_emission

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            # [B, T, T] --> [B, 1, T] --> [B, T]
            next_score, indices = next_score.max(dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            # ([S, B]-->[B]-->[B, 1]) + [B, T] + [B, T] --> [B, T]
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Now, compute the best path for each sample

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []
        best_score_list = []

        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            # ([B, T]-->[T]) --> ([1], [1])
            best_score, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            for hist in reversed(history[:seq_ends[idx]]):
                # [B, T] -->
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags_list.append(best_tags)
            best_score_list.append(best_score.item())

        return best_tags_list, best_score_list

    def _viterbi_multi_decode(self, emissions, mask, topk):
        """
        :param emissions: [S, B, T]
        :param mask: [S, B]
        :param topk: int
        :return:
        """
        seq_length, batch_size = mask.shape
        _, _, num_tags = emissions.shape
        if topk > num_tags:
            raise RuntimeError(
                'topk must be smaller than number of tags, however now topk is {}, number is {}'.format(topk, num_tags))

        """index=0"""
        # [T] + ([S, B, T]-->[B, T]) --> [B, T]
        score = self.start_transitions + emissions[0]

        """index=1"""
        history = []
        # [B, T] --> [B, T, 1]
        broadcast_score = score.unsqueeze(2)
        # [S, B, T] --> [B, T] --> [B, 1, T]
        broadcast_emission = emissions[1].unsqueeze(1)
        # [B, T, 1] + [T, T] + [B, 1, T] --> [B, T, T]
        next_score = broadcast_score + self.transitions + broadcast_emission
        # [B, T, T] --> [B, topk, T], [B, topK, T]
        next_score, indices = next_score.topk(topk, 1)  # 1: along the first dimension
        # ([S, B]-->[B]-->[B, 1]-->[B, 1, 1]) + [B, topk, T] + [B, 1, T] --> [B, topk, T]
        score = torch.where(mask[1].unsqueeze(1).unsqueeze(1), next_score, score.unsqueeze(1))
        # [B, topK, T]
        history.append(indices)

        """index>1"""
        for i in range(2, seq_length):
            # [B, topk, T] --> [B, topk, T, 1]
            broadcast_score = score.unsqueeze(3)
            # [S, B, T] --> [B, T] --> [B, 1, T] --> [B, 1, 1, T]
            broadcast_emission = emissions[i].unsqueeze(1).unsqueeze(2)
            # [B, topk, T, 1] + [T, T] + [B, 1, 1, T] --> [B, topk, T, T] --> [B, topk*T, T]
            next_score = (broadcast_score + self.transitions + broadcast_emission).view(batch_size, topk*num_tags, num_tags)
            # [B, topk*T, T] --> [B, topk, T], [B, topk, T]
            next_score, indices = next_score.topk(topk, 1)
            # ([S, B]-->[B]-->[B, 1]-->[B, 1, 1]) + [B, topk, T] + [B, topk, T] --> [B, topk, T]
            score = torch.where(mask[i].unsqueeze(1).unsqueeze(1), next_score, score)
            # [B, topk, T]
            history.append(indices)

        # [B, topk, T] + [T] --> [B, topk, T]
        score += self.end_transitions

        # [S, B] --> [B]
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []
        best_score_list = []

        for idx in range(batch_size):
            # [B, topk, T] --> [topk, T] --> [topk*T] --> ([topk], [topk])
            last_score, best_last_tag = score[idx].view(-1).topk(topk)
            best_score_list.append(last_score.tolist())
            # Tensor::[topk] --> [Tensor::[topk]]
            best_tags = [best_last_tag]

            for hist in reversed(history[:seq_ends[idx]]):
                # ([B, topk, T]-->[topk, T]-->[topk*T])[topk] --> [topk]
                best_last_tag = torch.take(hist[idx], best_tags[-1])
                best_tags.append(best_last_tag)

            # [Tensor::[topk]] --> [List::[topk]]
            best_tags = list(map(lambda _: (_ % num_tags).tolist(), best_tags))
            best_tags.reverse()
            best_tags_list.append(best_tags)

        # [B, S, topk]
        return best_tags_list, best_score_list

    @staticmethod
    def _build_transition_matrix(num_labels):
        """
        tag_num: 3, eg.[a, b, c]
        transition_matrix:
            [
                [0, 0, 0, 0, 0, -inf, -inf],  # B-a
                [0, 0, 0, 0, -inf, 0, -inf],  # B-b
                [0, 0, 0, 0, -inf, -inf, 0],  # B-c
                [0, 0, 0, 0, -inf, -inf, -inf],  # O
                [0, 0, 0, 0, 0, -inf, -inf],  # I-a
                [0, 0, 0, 0, -inf, 0, -inf],  # I-b
                [0, 0, 0, 0, -inf, -inf, 0]  # I-c
            ]
        """
        ret = []
        for i in range(num_labels):
            ret.append([0] * (num_labels + 1) + [float("-inf")] * i + [0] + [float("-inf")] * (num_labels - i - 1))
        ret.append([0] * (num_labels + 1) + [float("-inf")] * (num_labels))
        for i in range(num_labels):
            ret.append([0] * (num_labels + 1) + [float("-inf")] * i + [0] + [float("-inf")] * (num_labels - i - 1))
        return np.array(ret).astype('float32')


# crf = CRF(3, True)
# emissions = torch.rand([2, 4, 3])
# ret = crf.decode(emissions, topk=2)
