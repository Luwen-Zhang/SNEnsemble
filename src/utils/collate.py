import collections.abc
import torch
import re
from torch._six import string_classes
import torch.utils.data as Data
import pandas as pd
import numpy as np

default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}"
)

np_str_obj_array_pattern = re.compile(r"[SaUO]")


def fix_collate_fn(batch):
    """
    This function implements the collation of pd.DataFrame.
    The calculation of ``transposed=list(zip(*batch))`` in ``default_collate`` is modified due to the following unknown
    exception. I implement this function to test whether the internal ``list``, ``zip``, or ``*`` is the reason.
    However, whether this function is effective is under testing. Since it does not influence the results, I will keep
    it active until the error occurs again.

    Fatal Python error: Segmentation fault

    Current thread 0x00007f910a78b100 (most recent call first):
      File "/home/xlluo/anaconda3/envs/mlfatigue/lib/python3.8/site-packages/torch/utils/data/_utils/collate.py", line 172 in default_collate
      File "/home/xlluo/anaconda3/envs/mlfatigue/lib/python3.8/site-packages/torch/utils/data/_utils/fetch.py", line 52 in fetch
      File "/home/xlluo/anaconda3/envs/mlfatigue/lib/python3.8/site-packages/torch/utils/data/dataloader.py", line 692 in _next_data
      File "/home/xlluo/anaconda3/envs/mlfatigue/lib/python3.8/site-packages/torch/utils/data/dataloader.py", line 652 in __next__
      File "segfault_test.py", line 52 in <module>
    Segmentation fault (core dumped)
    """
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel, device=elem.device)
            out = elem.new(storage).resize_(len(batch), *list(elem.size()))
        return torch.stack(batch, 0, out=out)
    elif "pandas" in elem_type.__module__ and isinstance(elem, pd.DataFrame):
        return pd.concat(batch)
    elif "pandas" in elem_type.__module__ and isinstance(elem, pd.Series):
        return pd.DataFrame(
            columns=elem.index,
            index=np.arange(len(batch)),
            data=np.vstack([i.values for i in batch]),
        )
    elif isinstance(elem, Data.Subset):
        dataset = elem.dataset
        indices = np.concatenate([elem.indices for elem in batch])
        return Data.Subset(dataset, indices)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return fix_collate_fn([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type(
                {key: fix_collate_fn([d[key] for d in batch]) for key in elem}
            )
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: fix_collate_fn([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(fix_collate_fn(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = [
            tuple([batch[i][j] for i in range(len(batch))]) for j in range(len(elem))
        ]

        if isinstance(elem, tuple):
            return [
                fix_collate_fn(samples) for samples in transposed
            ]  # Backwards compatibility.
        else:
            try:
                return elem_type([fix_collate_fn(samples) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [fix_collate_fn(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))
