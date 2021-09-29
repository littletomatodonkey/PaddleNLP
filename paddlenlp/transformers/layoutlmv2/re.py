import copy

import paddle
from paddle import nn
from paddle.nn import CrossEntropyLoss


class BiaffineAttention(nn.Layer):
    """Implements a biaffine attention operator for binary relation classification.

    PyTorch implementation of the biaffine attention operator from "End-to-end neural relation
    extraction using deep biaffine attention" (https://arxiv.org/abs/1812.11275) which can be used
    as a classifier for binary relation classification.

    Args:
        in_features (int): The size of the feature dimension of the inputs.
        out_features (int): The size of the feature dimension of the output.

    Shape:
        - x_1: `(N, *, in_features)` where `N` is the batch dimension and `*` means any number of
          additional dimensisons.
        - x_2: `(N, *, in_features)`, where `N` is the batch dimension and `*` means any number of
          additional dimensions.
        - Output: `(N, *, out_features)`, where `N` is the batch dimension and `*` means any number
            of additional dimensions.

    Examples:
        >>> batch_size, in_features, out_features = 32, 100, 4
        >>> biaffine_attention = BiaffineAttention(in_features, out_features)
        >>> x_1 = torch.randn(batch_size, in_features)
        >>> x_2 = torch.randn(batch_size, in_features)
        >>> output = biaffine_attention(x_1, x_2)
        >>> print(output.size())
        torch.Size([32, 4])
    """

    def __init__(self, in_features, out_features):
        super(BiaffineAttention, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.bilinear = nn.Bilinear(
            in_features, in_features, out_features, bias_attr=False)
        self.linear = nn.Linear(2 * in_features, out_features)

    def forward(self, x_1, x_2):
        return self.bilinear(
            x_1, x_2) + self.linear(paddle.concat(
                (x_1, x_2), axis=-1))


class REDecoder(nn.Layer):
    def __init__(self, hidden_size=768, hidden_dropout_prob=0.1):
        super().__init__()
        # TODO scale_grad_by_freq=true
        self.entity_emb = nn.Embedding(3, hidden_size)
        projection = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(hidden_dropout_prob), )
        self.ffnn_head = copy.deepcopy(projection)
        self.ffnn_tail = copy.deepcopy(projection)
        self.rel_classifier = BiaffineAttention(hidden_size // 2, 2)
        self.loss_fct = CrossEntropyLoss()

    def build_relation(self, relations, entities):
        batch_size = len(relations)
        new_relations = []
        for b in range(batch_size):
            if len(entities[b]["start"]) <= 2:
                entities[b] = {"end": [1, 1], "label": [0, 0], "start": [0, 0]}
            all_possible_relations = set([
                (i, j)
                for i in range(len(entities[b]["label"]))
                for j in range(len(entities[b]["label"]))
                if entities[b]["label"][i] == 1 and entities[b]["label"][j] == 2
            ])
            if len(all_possible_relations) == 0:
                all_possible_relations = set([(0, 1)])
            positive_relations = set(
                list(zip(relations[b]["head"], relations[b]["tail"])))
            negative_relations = all_possible_relations - positive_relations
            positive_relations = set(
                [i for i in positive_relations if i in all_possible_relations])
            reordered_relations = list(positive_relations) + list(
                negative_relations)
            relation_per_doc = {"head": [], "tail": [], "label": []}
            relation_per_doc["head"] = [i[0] for i in reordered_relations]
            relation_per_doc["tail"] = [i[1] for i in reordered_relations]
            relation_per_doc["label"] = [1] * len(positive_relations) + [0] * (
                len(reordered_relations) - len(positive_relations))
            assert len(relation_per_doc["head"]) != 0
            new_relations.append(relation_per_doc)
        return new_relations, entities

    def get_predicted_relations(self, logits, relations, entities):
        pred_relations = []
        for i, pred_label in enumerate(logits.argmax(-1)):
            if pred_label != 1:
                continue
            rel = {}
            rel["head_id"] = relations["head"][i]
            rel["head"] = (entities["start"][rel["head_id"]],
                           entities["end"][rel["head_id"]])
            rel["head_type"] = entities["label"][rel["head_id"]]

            rel["tail_id"] = relations["tail"][i]
            rel["tail"] = (entities["start"][rel["tail_id"]],
                           entities["end"][rel["tail_id"]])
            rel["tail_type"] = entities["label"][rel["tail_id"]]
            rel["type"] = 1
            pred_relations.append(rel)
        return pred_relations

    def forward(self, hidden_states, entities, relations):
        batch_size, max_n_words, context_dim = hidden_states.shape
        relations, entities = self.build_relation(relations, entities)
        loss = 0
        all_pred_relations = []
        for b in range(batch_size):
            head_entities = paddle.to_tensor(relations[b]["head"])
            tail_entities = paddle.to_tensor(relations[b]["tail"])
            relation_labels = paddle.to_tensor(relations[b]["label"])
            entities_start_index = paddle.to_tensor(entities[b]["start"])
            entities_labels = paddle.to_tensor(entities[b]["label"])
            head_index = entities_start_index[head_entities]
            head_label = entities_labels[head_entities]
            head_label_repr = self.entity_emb(head_label)

            tail_index = entities_start_index[tail_entities]
            tail_label = entities_labels[tail_entities]
            tail_label_repr = self.entity_emb(tail_label)

            tmp_hidden_states = hidden_states[b][head_index]
            if len(tmp_hidden_states.shape) == 1:
                tmp_hidden_states = paddle.unsqueeze(tmp_hidden_states, axis=0)
            head_repr = paddle.concat(
                (tmp_hidden_states, head_label_repr), axis=-1)

            tmp_hidden_states = hidden_states[b][tail_index]
            if len(tmp_hidden_states.shape) == 1:
                tmp_hidden_states = paddle.unsqueeze(tmp_hidden_states, axis=0)
            tail_repr = paddle.concat(
                (tmp_hidden_states, tail_label_repr), axis=-1)

            heads = self.ffnn_head(head_repr)
            tails = self.ffnn_tail(tail_repr)
            logits = self.rel_classifier(heads, tails)
            loss += self.loss_fct(logits, relation_labels)
            pred_relations = self.get_predicted_relations(logits, relations[b],
                                                          entities[b])
            all_pred_relations.append(pred_relations)
        return loss, all_pred_relations
