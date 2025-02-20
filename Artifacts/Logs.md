NeoX Setup Logs

- Created venv (`python -m venv neoxolmo`) and installed neox via README
    - pip install -r requirements/requirements.txt
    - pip install -r requirements/requirements-wandb.txt
    - pip install -r requirements/requirements-flashattention.txt 
        - Crashed due to `ModuleNotFoundError: No module named 'wheel'`
            - Referred to https://github.com/Vaibhavs10/insanely-fast-whisper/issues/226 and ran `pip install wheel`
        - Crashed due to RuntimeError: The detected CUDA version (11.8) mismatches the version that was used to compile PyTorch (12.4). Please make sure to use the same CUDA versions.
            - As per nvidia-smi CUDA version is 12.4 but as per `nvcc --version` CUDA version is 11.8
            - Used ChatGPT and SO to debug - https://chatgpt.com/share/6762a685-aa28-8011-aed0-1ae3a9dd6501, https://stackoverflow.com/questions/53422407/different-cuda-versions-shown-by-nvcc-and-nvidia-smi (SO Better)
                - Top Comment - "nvidia-smi shows you the CUDA version that your driver supports. You have one of the recent 410.x drivers installed which support CUDA 10. The version the driver supports has nothing to do with the version you compile and link your program against. A driver that supports CUDA 10.0 will also be able to run an application that was built for CUDA 9.2…"
            - Ran `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
            - Again got same issue 
            - Retried with `pip install --upgrade --force-reinstall --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` to reinstall
                - Got some errors ```ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts. pydantic 2.10.3 requires typing-extensions>=4.12.2, but you typing-extensions 4.9.0 which is incompatible.``` but they seem ignorable
            - Reran flash attention install (Worked!)
    - Error when running pretraining run on 2xH100 node due to typing_extension import
        - Ran `pip install --upgrade typing-extensions`
- Made these changes when trying to tokenizer Wiki via GPT2 - https://github.com/microsoft/DeepSpeed/issues/5337 and https://github.com/microsoft/DeepSpeed/issues/5603 (also change log to logger)
    - Command - python prepare_data.py -d ./data
- Trying Wiki/Pile Shard with OLMo Tokenizer - 
    - Command - python prepare_data.py -d ./data -t HFTokenizer --vocab-file /NS/llm-pretraining/work/afkhan/USC_Colab/gpt-neox/Artifacts/olmo_tokenizer.json pile_subset
    - Does not work for Pile (server down)
    - Tried for Wiki - python prepare_data.py -d ./data-enwiki-olmo-tokenizer -t HFTokenizer --vocab-file /NS/llm-pretraining/work/afkhan/USC_Colab/gpt-neox/Artifacts/olmo_tokenizer.json
    - Worked and created separate folder for clean distinction
- Testing with Pythia 19M - 
    - Ran - `./deepy.py train.py ./configs/19M.yml ./configs/local_setup_wandb_modified.yml` (This ran with GPT2 Tokenizer)
    - Seems to work
- Testing with 1-3B.yml config 
    - ENV - `source /NS/venvs/work/afkhan/neoxolmo/bin/activate`
    - Ran - `./deepy.py train.py ./configs/1-3B.yml ./configs/local_setup_wandb_modified.yml` (This ran with GPT2 Tokenizer)
    - See [WANDB](https://wandb.ai/aflah/neox?nw=nwuseraflah)
- For MultiNode SLURM - 
    - https://github.com/EleutherAI/gpt-neox?tab=readme-ov-file#hostfile-generation
- Tokenize with OLMo 1 -
    - `python prepare_data.py -d ./data-enwiki-olmo-tokenizer -t HFTokenizer --vocab-file /NS/llm-pretraining/work/afkhan/USC_Colab/gpt-neox/Artifacts/olmo_tokenizer.json`
- Finally nailed down what hangs stuff - Setting scaled_upper_triang_masked_softmax_fusion to True
- Set hidden dim size to be whatever was present in llama2 config.
- When exporting set the intermediate size to what you wanted (like 11008) in config otherwise error
- Export command 6.7B - `python ./tools/ckpts/convert_neox_to_hf_UPDATED.py --input_dir /NS/llm-pretraining/work/afkhan/USC_Colab/gpt-neox/checkpoints-Hubble-6.7B-FA-BS-32-GAS-2-2x8xA100-PP-1-MP-2-OLMo-Tokenizer-Int-32768/global_step500 --config_file /NS/llm-pretraining/work/afkhan/USC_Colab/gpt-neox/configs/hubble/6_7B_export.yml --output_dir Artifacts/Exported_HF_Model/Hubble_6_7B/ --precision auto --architecture llama`
- Export command 1.1B - `python ./tools/ckpts/convert_neox_to_hf_UPDATED.py --input_dir /NS/llm-pretraining/work/afkhan/USC_Colab/gpt-neox/checkpoints-Hubble-1.1B-FA-BS-32-GAS-1-2x8xA100-PP-1-MP-1-OLMo-Tokenizer-Int-16896/global_step4 --config_file /NS/llm-pretraining/work/afkhan/USC_Colab/gpt-neox/configs/hubble/1_1B_export.yml --output_dir Artifacts/Exported_HF_Model/Hubble_1_1B/ --precision auto --architecture llama`
- Export command 2.8B - `python ./tools/ckpts/convert_neox_to_hf_UPDATED.py --input_dir /NS/llm-pretraining/work/afkhan/USC_Colab/gpt-neox/checkpoints-Hubble-2.8B-FA-BS-32-GAS-1-2x8xA100-PP-1-MP-1-OLMo-Tokenizer-Int-30720/global_step1 --config_file /NS/llm-pretraining/work/afkhan/USC_Colab/gpt-neox/configs/hubble/2_8B_export.yml --output_dir Artifacts/Exported_HF_Model/Hubble_2_8B_Step_1/ --precision auto --architecture llama`
- Export command 2.8B - `python ./tools/ckpts/convert_neox_to_hf_UPDATED.py --input_dir /NS/llm-pretraining/work/afkhan/USC_Colab/gpt-neox/checkpoints-Hubble-2.8B-FA-BS-32-GAS-1-2x8xA100-PP-1-MP-1-OLMo-Tokenizer-Int-24576/global_step1 --config_file /NS/llm-pretraining/work/afkhan/USC_Colab/gpt-neox/configs/hubble/2_8B_export.yml --output_dir Artifacts/Exported_HF_Model/Hubble_2_8B_Step_1_Int_24576/ --precision auto --architecture llama`
- Export command 1.1B GQA - `python ./tools/ckpts/convert_neox_to_hf_UPDATED.py --input_dir /NS/llm-pretraining/work/afkhan/USC_Colab/gpt-neox/Master_CKPTS/checkpoints-Hubble-1.1B-Baseline-BS-8-GAS-8-No-Activation-Checkpointing-A100-GQA-KV-Heads-4-All-Fusion/global_step1 --config_file /NS/llm-pretraining/work/afkhan/USC_Colab/gpt-neox/Master_CKPTS/checkpoints-Hubble-1.1B-Baseline-BS-8-GAS-8-No-Activation-Checkpointing-A100-GQA-KV-Heads-4-All-Fusion/global_step1/configs/1_1B_Baseline_BS_8_GAS_8_No_Activation_Checkpointing_GQA_KV_Heads_4_All_Fusion_Export.yml --output_dir Artifacts/Exported_HF_Model/1_1B_Baseline_BS_8_GAS_8_No_Activation_Checkpointing_GQA_KV_Heads_4_All_Fusion_Export/ --precision auto --architecture llama`
- Export command 6.7B GQA - `python ./tools/ckpts/convert_neox_to_hf_UPDATED.py --input_dir /NS/llm-pretraining/work/afkhan/USC_Colab/gpt-neox/Master_CKPTS/checkpoints-Hubble-6.7B-Baseline-No-Activation-Checkpointing-BS-4-GAS-16-GQA-KV-Heads-4-Both-Fusion/global_step1000 --config_file /NS/llm-pretraining/work/afkhan/USC_Colab/gpt-neox/Master_CKPTS/checkpoints-Hubble-6.7B-Baseline-No-Activation-Checkpointing-BS-4-GAS-16-GQA-KV-Heads-4-Both-Fusion/global_step1000/configs/6_7B_Baseline_No_Activation_Checkpointing_BS_4_GAS_16_GQA_KV_Heads_4_Both_Fusion_Export.yml --output_dir Artifacts/Exported_HF_Model/6_7B_Baseline_No_Activation_Checkpointing_BS_4_GAS_16_GQA_KV_Heads_4_Both_Fusion_Export_Step_1000/ --precision auto --architecture llama`
- If stuff hangs delete /home/afkhan/.cache/torch_extensions or torch_extensions build dir

Ryan's Steps - 

conda create -p hubble2 python=3.10
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements/requirements.txt
pip install triton=2.0.0
pip install -r requirements/requirements-wandb.txt
pip install -r ./requirements/requirements-flashattention.txt

CUDA 12.1 ENV - 

cuda_version=12.1
export PATH=/usr/lib/cuda-${cuda_version}/bin/:${PATH}
export LD_LIBRARY_PATH=/usr/lib/cuda-${cuda_version}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_PATH=/usr/lib/cuda-${cuda_version}/
- pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121 
- Above command borrowed from https://pytorch.org/get-started/previous-versions/#linux-and-windows-1
- pip install -r requirements/requirements.txt
- pip install -r requirements/requirements-wandb.txt
- pip install -r ./requirements/requirements-flashattention.txt
- pip install -r ./requirements/requirements-transformerengine.txt

