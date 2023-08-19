import torch
from torch import nn as nn
from transformers import BertConfig
from transformers import BertModel
from transformers import BertPreTrainedModel

from aspert import sampling
from aspert import util


def get_token(h: torch.tensor, x: torch.tensor, token: int):
    """ Get specific token embedding (e.g. [CLS]) """
    emb_size = h.shape[-1]

    token_h = h.view(-1, emb_size)
    flat = x.contiguous().view(-1)

    # get contextualized embedding of given token
    token_h = token_h[flat == token, :]

    return token_h


class ASpERT(BertPreTrainedModel):
    """ Span-based model to jointly extract entities and relations """

    VERSION = '1.1'

    def __init__(self, config: BertConfig, cls_token: int, relation_types: int, entity_types: int,
                 size_embedding: int, prop_drop: float, freeze_transformer: bool, max_pairs: int = 100):
        super(ASpERT, self).__init__(config)

        # BERT model
        self.bert = BertModel(config)

        # layers 784
        self.rel_classifier1 = nn.Linear(config.hidden_size * 3 + size_embedding * 2 + (config.num_attention_heads * config.num_hidden_layers) * 2, relation_types)
        # scierc:2642
        # self.rel_classifier2 = nn.Linear(784, relation_types)
        self.entity_classifier1 = nn.Linear(config.hidden_size * 2 + size_embedding + config.num_attention_heads * config.num_hidden_layers, 784)
        # scierc:1705
        self.entity_classifier2 = nn.Linear(784, entity_types)
        self.size_embeddings = nn.Embedding(100, size_embedding)
        self.dropout = nn.Dropout(prop_drop)
        # self.selu = nn.SELU()
        self.relu = nn.ReLU()

        self._cls_token = cls_token
        self._relation_types = relation_types
        self._entity_types = entity_types
        self._max_pairs = max_pairs

        # weight initialization
        self.init_weights()

        if freeze_transformer:
            print("Freeze transformer weights")

            # freeze all transformer weights
            for param in self.bert.parameters():
                param.requires_grad = False

    def _forward_train(self, encodings: torch.tensor, context_masks: torch.tensor, entity_masks: torch.tensor,
                       entity_sizes: torch.tensor, relations: torch.tensor, rel_masks: torch.tensor):
        # get contextualized token embeddings from last transformer layer
        context_masks = context_masks.float()
        output = self.bert(input_ids=encodings, attention_mask=context_masks, output_attentions=True)
        # (batch_size, sequence_length, hidden_size)
        h = output['last_hidden_state']
        # (batch_size, num_heads, sequence_length, sequence_length)
        a = output['attentions']
        a = torch.cat(a, 1)
        a = a.permute(0, 2, 3, 1)
        att = a.unsqueeze(1).repeat(1, entity_masks.shape[1], 1, 1, 1)
        # (batch_size, entity_num, sequence_length, sequence_length, num_heads)
        att = att + (entity_masks.unsqueeze(-1).unsqueeze(-1)==0).float() * (-1e30)
        # entity_masks (batch_size, entity_num, sequence_length, 1)

        batch_size = encodings.shape[0]

        # classify entities
        size_embeddings = self.size_embeddings(entity_sizes)  # embed entity candidate sizes
        entity_clf, entity_spans_pool, att_pool = self._classify_entities(encodings, h, att, entity_masks, size_embeddings)

        # classify relations
        h_large = h.unsqueeze(1).repeat(1, max(min(relations.shape[1], self._max_pairs), 1), 1, 1)
        rel_clf = torch.zeros([batch_size, relations.shape[1], self._relation_types]).to(
            self.rel_classifier1.weight.device)

        # obtain relation logits
        # chunk processing to reduce memory usage
        for i in range(0, relations.shape[1], self._max_pairs):
            # classify relation candidates
            chunk_rel_logits = self._classify_relations(entity_spans_pool, att_pool, size_embeddings,
                                                        relations, rel_masks, h_large, i)
            rel_clf[:, i:i + self._max_pairs, :] = chunk_rel_logits

        return entity_clf, rel_clf

    def _forward_inference(self, encodings: torch.tensor, context_masks: torch.tensor, entity_masks: torch.tensor,
                           entity_sizes: torch.tensor, entity_spans: torch.tensor, entity_sample_masks: torch.tensor):
        # get contextualized token embeddings from last transformer layer
        context_masks = context_masks.float()

        output = self.bert(input_ids=encodings, attention_mask=context_masks, output_attentions=True)
        h = output['last_hidden_state']
        a = output['attentions']
        a = torch.cat(a, 1)
        a = a.permute(0, 2, 3, 1)
        att = a.unsqueeze(1).repeat(1, entity_masks.shape[1], 1, 1, 1)
        # (batch_size, entity_num, sequence_length, sequence_length, num_heads)
        masks = (entity_masks.unsqueeze(-1).unsqueeze(-1) == 0).float() * (-1e30)
        for i in range(0, att.shape[1], 100):
            # classify relation candidates
            chunk_entity_att = att[:, i:i + 100, :, :, :] + masks[:, i:i + 100, :, :, :]
            att[:, i:i + 100, :, :, :] = chunk_entity_att

        # entity_masks (batch_size, entity_num, sequence_length, 1, 1)

        batch_size = encodings.shape[0]
        ctx_size = context_masks.shape[-1]

        # classify entities
        size_embeddings = self.size_embeddings(entity_sizes)  # embed entity candidate sizes
        entity_clf, entity_spans_pool, att_pool = self._classify_entities(encodings, h, att, entity_masks, size_embeddings)

        # ignore entity candidates that do not constitute an actual entity for relations (based on classifier)
        relations, rel_masks, rel_sample_masks = self._filter_spans(entity_clf, entity_spans,
                                                                    entity_sample_masks, ctx_size)

        rel_sample_masks = rel_sample_masks.float().unsqueeze(-1)
        h_large = h.unsqueeze(1).repeat(1, max(min(relations.shape[1], self._max_pairs), 1), 1, 1)
        rel_clf = torch.zeros([batch_size, relations.shape[1], self._relation_types]).to(
            self.rel_classifier1.weight.device)

        # obtain relation logits
        # chunk processing to reduce memory usage
        for i in range(0, relations.shape[1], self._max_pairs):
            # classify relation candidates
            chunk_rel_logits = self._classify_relations(entity_spans_pool, att_pool, size_embeddings,
                                                        relations, rel_masks, h_large, i)
            # apply sigmoid
            chunk_rel_clf = torch.sigmoid(chunk_rel_logits)
            rel_clf[:, i:i + self._max_pairs, :] = chunk_rel_clf

        rel_clf = rel_clf * rel_sample_masks  # mask

        # apply softmax
        entity_clf = torch.softmax(entity_clf, dim=2)

        return entity_clf, rel_clf, relations

    def _classify_entities(self, encodings, h, att, entity_masks, size_embeddings):
        # max pool entity candidate spans
        # att (batch_size, entity_num, layer_num_heads)
        # att_pool = att.mean(dim=3).max(dim=2)[0]
        # att_pool = att.max(dim=3)[0].max(dim=2)[0]
        '''
        添加阈值
        '''
        # (batch_size, entity_num, sequence_length, sequence_length, layer_num_heads)
        theta = 0.5
        att_gt = torch.gt(att,theta)

        for i in range(0, att.shape[1], 50):
            # chunk_att_pool = (att[:, i:i + 50, :, :, :] * att_gt[:, i:i + 50, :, :, :]).max(dim=3)[0].max(dim=2)[0]
            # chunk_att_pool = (att[:, i:i + 50, :, :, :] * att_gt[:, i:i + 50, :, :, :]).sum(dim=3).sum(dim=2)
            chunk_att_pool = (att[:, i:i + 50, :, :, :] * att_gt[:, i:i + 50, :, :, :]).mean(dim=3).mean(dim=2)
            if i == 0:
                att_pool = chunk_att_pool
            else:
                att_pool = torch.cat([att_pool,chunk_att_pool],dim=1)

        m = (entity_masks.unsqueeze(-1) == 0).float() * (-1e30)
        entity_spans_pool = m + h.unsqueeze(1).repeat(1, entity_masks.shape[1], 1, 1)
        entity_spans_pool = entity_spans_pool.max(dim=2)[0]

        # get cls token as candidate context representation
        entity_ctx = get_token(h, encodings, self._cls_token)

        # create candidate representations including context, max pooled span and size embedding
        entity_repr = torch.cat([entity_ctx.unsqueeze(1).repeat(1, entity_spans_pool.shape[1], 1),
                                 entity_spans_pool, size_embeddings, att_pool], dim=2)
        entity_repr = self.dropout(entity_repr)

        # classify entity candidates
        entity_clf = self.relu(self.entity_classifier1(entity_repr))
        entity_clf = self.entity_classifier2(entity_clf)

        return entity_clf, entity_spans_pool, att_pool

    def _classify_relations(self, entity_spans, att_spans, size_embeddings, relations, rel_masks, h, chunk_start):
        batch_size = relations.shape[0]

        # create chunks if necessary
        if relations.shape[1] > self._max_pairs:
            relations = relations[:, chunk_start:chunk_start + self._max_pairs]
            rel_masks = rel_masks[:, chunk_start:chunk_start + self._max_pairs]
            h = h[:, :relations.shape[1], :]

        # get pairs of entity candidate representations
        entity_pairs = util.batch_index(entity_spans, relations)
        entity_pairs = entity_pairs.view(batch_size, entity_pairs.shape[1], -1)

        # att (batch_size, entity_num, layer_num_heads)

        att_entity_pairs = util.batch_index(att_spans, relations)
        att_entity_pairs = att_entity_pairs.view(batch_size, att_entity_pairs.shape[1], -1)

        # get corresponding size embeddings
        size_pair_embeddings = util.batch_index(size_embeddings, relations)
        size_pair_embeddings = size_pair_embeddings.view(batch_size, size_pair_embeddings.shape[1], -1)

        # relation context (context between entity candidate pair)
        # mask non entity candidate tokens
        m = ((rel_masks == 0).float() * (-1e30)).unsqueeze(-1)
        rel_ctx = m + h
        # max pooling
        rel_ctx = rel_ctx.max(dim=2)[0]
        # set the context vector of neighboring or adjacent entity candidates to zero
        rel_ctx[rel_masks.to(torch.uint8).any(-1) == 0] = 0

        # create relation candidate representations including context, max pooled entity candidate pairs
        # and corresponding size embeddings
        rel_repr = torch.cat([rel_ctx, entity_pairs, size_pair_embeddings, att_entity_pairs], dim=2)
        rel_repr = self.dropout(rel_repr)

        # classify relation candidates
        chunk_rel_logits = self.rel_classifier1(rel_repr)
        # chunk_rel_logits = self.relu(self.rel_classifier1(rel_repr))
        # chunk_rel_logits = self.rel_classifier2(chunk_rel_logits)
        return chunk_rel_logits

    def _filter_spans(self, entity_clf, entity_spans, entity_sample_masks, ctx_size):
        batch_size = entity_clf.shape[0]
        entity_logits_max = entity_clf.argmax(dim=-1) * entity_sample_masks.long()  # get entity type (including none)
        batch_relations = []
        batch_rel_masks = []
        batch_rel_sample_masks = []

        for i in range(batch_size):
            rels = []
            rel_masks = []
            sample_masks = []

            # get spans classified as entities
            non_zero_indices = (entity_logits_max[i] != 0).nonzero(as_tuple=False).view(-1)
            non_zero_spans = entity_spans[i][non_zero_indices].tolist()
            non_zero_indices = non_zero_indices.tolist()

            # create relations and masks
            for i1, s1 in zip(non_zero_indices, non_zero_spans):
                for i2, s2 in zip(non_zero_indices, non_zero_spans):
                    if i1 != i2:
                        rels.append((i1, i2))
                        rel_masks.append(sampling.create_rel_mask(s1, s2, ctx_size))
                        sample_masks.append(1)

            if not rels:
                # case: no more than two spans classified as entities
                batch_relations.append(torch.tensor([[0, 0]], dtype=torch.long))
                batch_rel_masks.append(torch.tensor([[0] * ctx_size], dtype=torch.bool))
                batch_rel_sample_masks.append(torch.tensor([0], dtype=torch.bool))
            else:
                # case: more than two spans classified as entities
                batch_relations.append(torch.tensor(rels, dtype=torch.long))
                batch_rel_masks.append(torch.stack(rel_masks))
                batch_rel_sample_masks.append(torch.tensor(sample_masks, dtype=torch.bool))

        # stack
        device = self.rel_classifier1.weight.device
        batch_relations = util.padded_stack(batch_relations).to(device)
        batch_rel_masks = util.padded_stack(batch_rel_masks).to(device)
        batch_rel_sample_masks = util.padded_stack(batch_rel_sample_masks).to(device)

        return batch_relations, batch_rel_masks, batch_rel_sample_masks

    def forward(self, *args, inference=False, **kwargs):
        if not inference:
            return self._forward_train(*args, **kwargs)
        else:
            return self._forward_inference(*args, **kwargs)


# Model access

_MODELS = {
    'aspert': ASpERT,
}


def get_model(name):
    return _MODELS[name]