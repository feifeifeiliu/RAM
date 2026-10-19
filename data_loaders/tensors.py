import torch

def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
    

def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['inp'] for b in notnone_batches]
    if 'lengths' in notnone_batches[0]:
        lenbatch = [b['lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]


    databatchTensor = collate_tensors(databatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting

    motion = databatchTensor
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor}}

    if 'text' in notnone_batches[0]:
        textbatch = [b['text'] for b in notnone_batches]
        cond['y'].update({'text': textbatch})

    if 'tokens' in notnone_batches[0]:
        textbatch = [b['tokens'] for b in notnone_batches]
        cond['y'].update({'tokens': textbatch})

    if 'action' in notnone_batches[0]:
        actionbatch = [b['action'] for b in notnone_batches]
        cond['y'].update({'action': torch.as_tensor(actionbatch).unsqueeze(1)})

    # collate action textual names
    if 'action_text' in notnone_batches[0]:
        action_text = [b['action_text']for b in notnone_batches]
        cond['y'].update({'action_text': action_text})
    
    if 'prefix' in notnone_batches[0]:
        cond['y'].update({'prefix': collate_tensors([b['prefix'] for b in notnone_batches])})
    
    if 'orig_lengths' in notnone_batches[0]:
        cond['y'].update({'orig_lengths': torch.as_tensor([b['orig_lengths'] for b in notnone_batches])})

    if 'key' in notnone_batches[0]:
        cond['y'].update({'db_key': [b['key'] for b in notnone_batches]})
        
    if 'file_id' in notnone_batches[0]:
        cond['y'].update({'file_id': [b['file_id'] for b in notnone_batches]})
        
    if 'tmr_emb' in notnone_batches[0]:
        cond['y'].update({'tmr_emb': collate_tensors([b['tmr_emb'] for b in notnone_batches])})

    return motion, cond

# an adapter to our collate func
def t2m_collate(batch, target_batch_size):
    repeat_factor = -(-target_batch_size // len(batch))  # Ceiling division
    repeated_batch = batch * repeat_factor 
    full_batch = repeated_batch[:target_batch_size]  # Truncate to the target batch size
    # batch.sort(key=lambda x: x[3], reverse=True)
    if full_batch[0].__len__() >= 7:
        adapted_batch = [{
            'inp': torch.tensor(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
            'text': b[2], #b[0]['caption']
            'tokens': b[6],
            'lengths': b[5],
            'key': b[7] if len(b) > 7 else None, # 我这里key是用作表示data source，0代表motion-x++，1代表humanml3d
            # 'file_id': b[0],
            # 'text_embed': b[8] if len(b) > 8 else None, 
        } for b in full_batch]
    else:
        adapted_batch = [{
            'inp': torch.tensor(b[2].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
            'text': b[1], #b[0]['caption']
            'tmr_emb': b[0],
            'lengths': b[3],
            'key': b[5], # 我这里key是用作表示data source，0代表motion-x++，1代表humanml3d
        } for b in full_batch]
    return collate(adapted_batch)


def t2m_prefix_collate(batch, pred_len):
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [{
        'inp': torch.tensor(b[4].T).float().unsqueeze(1)[..., -pred_len:], # [seqlen, J] -> [J, 1, seqlen]
        'prefix': torch.tensor(b[4].T).float().unsqueeze(1)[..., :-pred_len],
        'text': b[2], #b[0]['caption']
        'tokens': b[6],
        'lengths': pred_len,  # b[5],
        'orig_lengths': b[5][0], #  For evaluation
        'key': b[7] if len(b) > 7 else None,
    } for b in batch]
    return collate(adapted_batch)



def collate_contrastive(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['m_feature'] for b in notnone_batches]
    if 'm_lengths' in notnone_batches[0]:
        lenbatch = [b['m_lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['m_feature'][0]) for b in notnone_batches]

    databatchTensor = collate_tensors(databatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[1]).unsqueeze(-1)

    motion = databatchTensor
    cond = {'y': {'m_mask': maskbatchTensor, 'm_lengths': lenbatchTensor}}
    
    if 't_feature' in notnone_batches[0]:
        t_feature = collate_tensors([b['t_feature'] for b in notnone_batches])
        cond['y'].update({'t_feature': t_feature})
        
    if 't_lengths' in notnone_batches[0]:
        t_lenbatchTensor = torch.as_tensor([b['t_lengths'] for b in notnone_batches])
        t_maskbatchTensor = lengths_to_mask(t_lenbatchTensor, t_feature.shape[1]).unsqueeze(-1)
        cond['y'].update({'t_mask': t_maskbatchTensor})

    if 'sent_emb' in notnone_batches[0]:
        cond['y'].update({'sent_emb': collate_tensors([b['sent_emb'] for b in notnone_batches])})
        
    if 'clip_emb' in notnone_batches[0]:
        cond['y'].update({'clip_emb': collate_tensors([torch.from_numpy(b['clip_emb']) for b in notnone_batches])})

    if 'text' in notnone_batches[0]:
        textbatch = [b['text'] for b in notnone_batches]
        cond['y'].update({'text': textbatch})

    return motion, cond

# an adapter to our collate func
def t2m_collate_contrastive(batch):
    # text_x_dict['x'], text_x_dict['length'], sent_emb, caption, motion, length
    
    adapted_batch = [{
        't_feature': b[0],
        't_lengths': b[1],
        'sent_emb': b[2],
        'clip_emb': b[3],
        'text': b[4],
        'm_feature': torch.tensor(b[5]).float(),
        'm_lengths': b[6],
    } for b in batch]
    return collate_contrastive(adapted_batch)