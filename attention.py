import torch
from transformers import BertModel



def get_entity_attention(dataloader):

    model = BertModel.from_pretrained(
        pretrained_model_name_or_path='bert-base-cased',
        output_hidden_states=True,
        output_attentions=True)

    p = []
    for batch in dataloader:
        # all_attention : layer_num  * (batch_size, num_heads, sequence_length, sequence_length)
        all_attention = model(batch['encodings'], attention_mask=batch['context_masks'])['attentions']
        all_attention = torch.cat(all_attention, 1)

        pos_entity_masks = batch['pos_entity_masks']
        batch_size = pos_entity_masks.shape[0]
        entity_num = pos_entity_masks.shape[1]
        # squence_length = pos_entity_masks.shape[2]

        pos_entity_idx = (pos_entity_masks == 1).nonzero()
        entity_attention = torch.zeros((batch_size, entity_num, entity_num))
        for head_index in pos_entity_idx:
            for tail_index in pos_entity_idx:
                # 属于同一个句子的不同entity的token
                if head_index[0] == tail_index[0] and head_index[1] != tail_index[1]:
                    # 一个句子两个entity的token之间在各个层的头的softmax后的注意力
                    f = all_attention[head_index[0]:head_index[0] + 1, :, head_index[2]:head_index[2] + 1,
                        tail_index[2]:tail_index[2] + 1]
                    f = f.mean(1)[0][0][0]
                    entity_attention[head_index[0]][head_index[1]][tail_index[1]] += f


        p.append(entity_attention)
    # p = torch.tensor(p)
    torch.save(p, 'entity_attention.pt')
    print('记录完成！')
















