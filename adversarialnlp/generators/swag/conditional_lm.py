import json
import logging
from typing import Union, List, Dict, Any
import warnings

import torch
from torch.nn.modules import Dropout

import numpy
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.common.checks import ConfigurationError
from allennlp.common import Params
from allennlp.common.util import lazy_groups_of
from allennlp.modules.elmo_lstm import ElmoLstm
from allennlp.modules.highway import Highway
from allennlp.modules.scalar_mix import ScalarMix
from allennlp.nn.util import remove_sentence_boundaries, add_sentence_boundary_token_ids, get_device_of
from allennlp.data.token_indexers.elmo_indexer import ELMoCharacterMapper, ELMoTokenCharactersIndexer
from allennlp.data.dataset import Batch
from allennlp.data import Token, Vocabulary, Instance
from allennlp.data.fields import TextField


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# pylint: disable=attribute-defined-outside-init
class ConditionalLM(torch.nn.Module):
    """
    Run a language model on conditional inputs.
    Parameters
    ----------
    options_file : ``str``
        ELMo JSON options file
    weight_file : ``str``
        ELMo hdf5 weight file
    requires_grad: ``bool``, optional
        If True, compute gradient of ELMo parameters for fine tuning.
    vocab_to_cache : ``List[str]``, optional, (default = 0.5).
        A list of words to pre-compute and cache character convolutions
        for. If you use this option, _ElmoBiLm expects that you pass word
        indices of shape (batch_size, timesteps) to forward, instead
        of character indices. If you use this option and pass a word which
        wasn't pre-cached, this will break.
    """
    def __init__(self,
                 options_file: str,
                 weight_file: str,
                 requires_grad: bool = False,
                 vocab_to_cache: List[str] = None) -> None:
        super(ConditionalLM, self).__init__()

        self._token_embedder = _ElmoCharacterEncoder(options_file, weight_file, requires_grad=requires_grad)

        self._requires_grad = requires_grad
        if requires_grad and vocab_to_cache:
            logging.warning("You are fine tuning ELMo and caching char CNN word vectors. "
                            "This behaviour is not guaranteed to be well defined, particularly. "
                            "if not all of your inputs will occur in the vocabulary cache.")
        # This is an embedding, used to look up cached
        # word vectors built from character level cnn embeddings.
        self._word_embedding = None
        self._bos_embedding: torch.Tensor = None
        self._eos_embedding: torch.Tensor = None
        if vocab_to_cache:
            logging.info("Caching character cnn layers for words in vocabulary.")
            # This sets 3 attributes, _word_embedding, _bos_embedding and _eos_embedding.
            # They are set in the method so they can be accessed from outside the
            # constructor.
            self.create_cached_cnn_embeddings(vocab_to_cache)

        with open(cached_path(options_file), 'r') as fin:
            options = json.load(fin)
        if not options['lstm'].get('use_skip_connections'):
            raise ConfigurationError('We only support pretrained biLMs with residual connections')
        self._elmo_lstm = ElmoLstm(input_size=options['lstm']['projection_dim'],
                                   hidden_size=options['lstm']['projection_dim'],
                                   cell_size=options['lstm']['dim'],
                                   num_layers=options['lstm']['n_layers'],
                                   memory_cell_clip_value=options['lstm']['cell_clip'],
                                   state_projection_clip_value=options['lstm']['proj_clip'],
                                   requires_grad=requires_grad)
        self._elmo_lstm.load_weights(weight_file)
        # Number of representation layers including context independent layer
        self.num_layers = options['lstm']['n_layers'] + 1

    def get_output_dim(self):
        return 2 * self._token_embedder.get_output_dim()

    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                word_inputs: torch.Tensor = None) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Parameters
        ----------
        inputs: ``torch.Tensor``, required.
            Shape ``(batch_size, timesteps, 50)`` of character ids representing the current batch.
        word_inputs : ``torch.Tensor``, required.
            If you passed a cached vocab, you can in addition pass a tensor of shape ``(batch_size, timesteps)``,
            which represent word ids which have been pre-cached.
        Returns
        -------
        Dict with keys:
        ``'activations'``: ``List[torch.Tensor]``
            A list of activations at each layer of the network, each of shape
            ``(batch_size, timesteps + 2, embedding_dim)``
        ``'mask'``:  ``torch.Tensor``
            Shape ``(batch_size, timesteps + 2)`` long tensor with sequence mask.
        Note that the output tensors all include additional special begin and end of sequence
        markers.
        """
        if self._word_embedding is not None and word_inputs is not None:
            try:
                mask_without_bos_eos = (word_inputs > 0).long()
                # The character cnn part is cached - just look it up.
                embedded_inputs = self._word_embedding(word_inputs) # type: ignore
                # shape (batch_size, timesteps + 2, embedding_dim)
                type_representation, mask = add_sentence_boundary_token_ids(
                        embedded_inputs,
                        mask_without_bos_eos,
                        self._bos_embedding,
                        self._eos_embedding
                )
            except RuntimeError:
                # Back off to running the character convolutions,
                # as we might not have the words in the cache.
                token_embedding = self._token_embedder(inputs)
                mask = token_embedding['mask']
                type_representation = token_embedding['token_embedding']
        else:
            token_embedding = self._token_embedder(inputs)
            mask = token_embedding['mask']
            type_representation = token_embedding['token_embedding']
        lstm_outputs = self._elmo_lstm(type_representation, mask)

        # Prepare the output.  The first layer is duplicated.
        # Because of minor differences in how masking is applied depending
        # on whether the char cnn layers are cached, we'll be defensive and
        # multiply by the mask here. It's not strictly necessary, as the
        # mask passed on is correct, but the values in the padded areas
        # of the char cnn representations can change.
        output_tensors = [
                torch.cat([type_representation, type_representation], dim=-1) * mask.float().unsqueeze(-1)
        ]
        for layer_activations in torch.chunk(lstm_outputs, lstm_outputs.size(0), dim=0):
            output_tensors.append(layer_activations.squeeze(0))

        return {
                'activations': output_tensors,
                'mask': mask,
        }

    def create_cached_cnn_embeddings(self, tokens: List[str]) -> None:
        """
        Given a list of tokens, this method precomputes word representations
        by running just the character convolutions and highway layers of elmo,
        essentially creating uncontextual word vectors. On subsequent forward passes,
        the word ids are looked up from an embedding, rather than being computed on
        the fly via the CNN encoder.
        This function sets 3 attributes:
        _word_embedding : ``torch.Tensor``
            The word embedding for each word in the tokens passed to this method.
        _bos_embedding : ``torch.Tensor``
            The embedding for the BOS token.
        _eos_embedding : ``torch.Tensor``
            The embedding for the EOS token.
        Parameters
        ----------
        tokens : ``List[str]``, required.
            A list of tokens to precompute character convolutions for.
        """
        tokens = [ELMoCharacterMapper.bos_token, ELMoCharacterMapper.eos_token] + tokens
        timesteps = 32
        batch_size = 32
        chunked_tokens = lazy_groups_of(iter(tokens), timesteps)

        all_embeddings = []
        device = get_device_of(next(self.parameters()))
        for batch in lazy_groups_of(chunked_tokens, batch_size):
            # Shape (batch_size, timesteps, 50)
            batched_tensor = batch_to_ids(batch)
            # NOTE: This device check is for when a user calls this method having
            # already placed the model on a device. If this is called in the
            # constructor, it will probably happen on the CPU. This isn't too bad,
            # because it's only a few convolutions and will likely be very fast.
            if device >= 0:
                batched_tensor = batched_tensor.cuda(device)
            output = self._token_embedder(batched_tensor)
            token_embedding = output["token_embedding"]
            mask = output["mask"]
            token_embedding, _ = remove_sentence_boundaries(token_embedding, mask)
            all_embeddings.append(token_embedding.view(-1, token_embedding.size(-1)))
        full_embedding = torch.cat(all_embeddings, 0)

        # We might have some trailing embeddings from padding in the batch, so
        # we clip the embedding and lookup to the right size.
        full_embedding = full_embedding[:len(tokens), :]
        embedding = full_embedding[2:len(tokens), :]
        vocab_size, embedding_dim = list(embedding.size())

        from allennlp.modules.token_embedders import Embedding # type: ignore
        self._bos_embedding = full_embedding[0, :]
        self._eos_embedding = full_embedding[1, :]
        self._word_embedding = Embedding(vocab_size, # type: ignore
                                         embedding_dim,
                                         weight=embedding.data,
                                         trainable=self._requires_grad,
                                         padding_index=0)
