import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    inputs=[torch.tensor(item["input"],dtype=torch.float32).T for item in batch]
    targets=[torch.tensor(item["target"],dtype=torch.float32).T for item in batch]
    lengths=torch.tensor([x.shape[0] for x in inputs],dtype=torch.long,)
    inputs=pad_sequence(inputs,batch_first=True,)
    targets=pad_sequence(targets,batch_first=True,)
    inputs=inputs.permute(0,2,1)
    targets=targets.permute(0,2,1)
    return {"input":inputs,"target":targets,"lengths":lengths,"speaker":[item["speaker"] for item in batch],"sentence":[item["sentence"] for item in batch], "noise_type":batch[0]["noise_type"],"snr":batch[0]["snr"],}
