import dataclasses
import os, shutil
from tqdm import tqdm
import numpy as np

import transformers
import flexgen.flex_opt as flexgen
from flexgen.opt_config import disable_torch_init, restore_torch_init
import langchain.llms as llms

# i copied from upstream and added:
# - kwparams for ram-limited systems to use accelerate
# - short circuiting if the path exists (redundant upstream)
# - code to free space if tight
def download_opt_weights(model_name, path, **kwparams):
    """Download weights from huggingface."""
    import torch
    from transformers import OPTForCausalLM, BloomForCausalLM

    if "/" in model_name:
        model_name = model_name.split("/")[1].lower()
    path = os.path.join(path, f"{model_name}-np")
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.exists(path):
        return

    if "opt" in model_name:
        hf_model_name = "facebook/" + model_name
        model_class = OPTForCausalLM
    elif "bloom" in model_name:
        hf_model_name = "bigscience/" + model_name
        model_class = BloomForCausalLM
    else:
        raise ValueError("Invalid model name: {model_name}")

    print(f"Load the pre-trained pytorch weights of {model_name} from huggingface. "
          f"The downloading and cpu loading can take dozens of minutes. "
          f"If it seems to get stuck, you can monitor the progress by "
          f"checking the memory usage of this process.")

    disable_torch_init()
    model = model_class.from_pretrained(hf_model_name, torch_dtype=torch.float16,
                                        _fast_init=True,
                                        **kwparams
    )
    restore_torch_init()

    os.makedirs(path, exist_ok=True)

    # added: free disk space if needed
    needed_space = sum([np.prod(np.array(p.shape)) for p in model.parameters()]) * 2
    statvfs = os.statvfs(path)
    available_space = statvfs.f_frsize * statvfs.f_bavail
    if available_space < needed_space:
        print('Wiping transformers cache to free space.')
        shutil.rmtree(transformers.utils.TRANSFORMERS_CACHE)

    print(f"Convert the weights to numpy format under {path} ...")
    if "opt" in model_name:
        for name, param in tqdm(list(model.model.named_parameters())):
            name = name.replace("decoder.final_layer_norm", "decoder.layer_norm")
            param_path = os.path.join(path, name)
            with open(param_path, "wb") as f:
                np.save(f, param.cpu().detach().numpy())
    elif "bloom" in model_name:
        for name, param in tqdm(list(model.transformer.named_parameters())):
            param_path = os.path.join(path, name)
            with open(param_path, "wb") as f:
                np.save(f, param.cpu().detach().numpy())
    else:
        raise ValueError("Invalid model name: {model_name}")

class FlexGenLLM:
    def __init__(
            self,
            name = 'facebook/opt-6.7b',
            task = 'text-generation',
            tokenizer = None,
            env = None,
            opt_config = None,
            offload_policy = None,
            path = 'weights',
            model_kwargs = {},
            **flat_kwargs,
    ):
        self.name = name
        self.task = task
        self.tokenizer = tokenizer
        self.env = env
        self.opt_config = opt_config
        self.offload_policy = offload_policy
        self.path = path
        self.model_kwargs = {**model_kwargs, **flat_kwargs}
    def __enter__(self):
        owner, shortname = self.name.split('/')
        download_opt_weights(self.name, self.path, **self.model_kwargs)
        if self.tokenizer is None:
            self.tokenizer = self.name
        if isinstance(self.tokenizer, str):
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.tokenizer, **self.model_kwargs)
        if self.env is None:
            gpu = flexgen.TorchDevice('cuda:0')
            cpu = flexgen.TorchDevice('cpu')
            disk = flexgen.TorchDisk('flexgen_offload')
            self.env = flexgen.Env(
                gpu=gpu,
                cpu=cpu,
                disk=disk,
                mixed=flexgen.TorchMixedDevice([gpu, cpu, disk]))
        if self.offload_policy is None:
            self.offload_policy = flexgen.Policy(1, 1,
                    100, 0,
                    100, 0,
                    100, 0,
                    overlap=True, sep_layer=True, pin_weight=True,
                    cpu_cache_compute=False, attn_sparsity=1.0,
                    compress_weight=False,
                    comp_weight_config=flexgen.CompressionConfig(
                        num_bits=4, group_size=64,
                        group_dim=0, symmetric=False),
                    compress_cache=False,
                    comp_cache_config=flexgen.CompressionConfig(
                        num_bits=4, group_size=64,
                        group_dim=2, symmetric=False))
        if self.opt_config is None:
            self.opt_config = shortname
        if isinstance(self.opt_config, str):
            self.opt_config = flexgen.get_opt_config(self.opt_config)
        if self.opt_config.name != shortname:
            self.opt_config = dataclasses.replace(self.opt_config, name = shortname)
        self.model = flexgen.OptLM(self.opt_config, self.env, self.path, self.offload_policy)
        self.model.init_all_weights()
        self.pipeline = transformers.pipeline(
            task = self.task,
            model = self.model,
            tokenizer = self.tokenizer,
            model_kwargs = self.model_kwargs,
        )
        self.llm = llms.HuggingFacePipeline(
           pipeline = self.pipeline,
           model_id = self.name,
           model_kwargs = self.model_kwargs,
        )
        return self.llm
    def __exit__(self, *params):
        del self.llm
        del self.pipeline
        self.model.delete_all_weights()
        self.model.env.disk.close_copy_threads()
        del self.model
