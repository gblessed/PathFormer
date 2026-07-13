# WiFo
B. Liu, S. Gao, X. Liu, X. Cheng, L. Yang. “WiFo: Wireless Foundation Model for Channel Prediction. ” SCIENCE CHINA Information Sciences, June 2025, 68(6): 162302. 
<br>

## 📢 Latest Updates

🔥🔥🔥 Last Updated on 2025.11.27 🔥🔥🔥 
- [2025.11.28] We released all the pre-training datasets as open source at [[Pretraining Dataset]](https://disk.pku.edu.cn/link/AA003E48DD5EF343C18ACD92ACF3BB8E3E).

## Dependencies and Installation
- Python 3.9 (Recommend to use [Anaconda](https://www.anaconda.com/))
- Pytorch 2.0.0
- NVIDIA GPU + CUDA
- Python packages: `pip install -r requirements.txt`


## Dataset Preparation
The dataset used for inference is available in [[Testing Dataset]](https://huggingface.co/datasets/liuboxun/WiFo-dataset).
The dataset used for pre-training is available in [[Pretraining Dataset]](https://disk.pku.edu.cn/link/AA003E48DD5EF343C18ACD92ACF3BB8E3E)


## Get Started
We have released the pre-trained weights of WiFo-Large/Base/Small/Little/Tiny for inference in [[Model](https://huggingface.co/liuboxun/WiFo)].

### Inference command for multi-dataset unified learning
```
python main.py --device_id 1 --size base --mask_strategy_random none --mask_strategy temporal --dataset D1*D2*D3*D4*D5*D6*D7*D8*D9*D10*D11*D12*D13*D14*D15*D16 --file_load_path ./weights/wifo_base --few_ratio 0.0 --t_patch_size 4 --patch_size 4 --batch_size 128 --pos_emb SinCos_3D
```
```
python main.py --device_id 1 --size base --mask_strategy_random none --mask_strategy fre --dataset D1*D2*D3*D4*D5*D6*D7*D8*D9*D10*D11*D12*D13*D14*D15*D16 --file_load_path ./weights/wifo_base --few_ratio 0.0 --t_patch_size 4 --patch_size 4 --batch_size 128 --pos_emb SinCos_3D
```
### Inference command for zero-shot generalization
```
python main.py --device_id 0 --size base --mask_strategy_random none --mask_strategy fre --dataset D17 --file_load_path ./weights/wifo_base --few_ratio 0.0 --t_patch_size 4 --patch_size 4 --batch_size 128 --pos_emb SinCos_3D
```

### Inference command for My Dataset
```
python main.py --device_id 0 --size base --mask_strategy_random none --mask_strategy fre --dataset city_47_chicago_3p5 --file_load_path ./weights/wifo_base --few_ratio 0.0 --t_patch_size 4 --patch_size 4 --batch_size 128 --pos_emb SinCos_3D


python main.py --device_id 0 --size base --mask_strategy_random none --mask_strategy fre --dataset all_scenarios --file_load_path ./weights/wifo_base --few_ratio 0.0 --t_patch_size 4 --patch_size 4 --batch_size 128 --pos_emb SinCos_3D

python main.py --my_data True --dataset all --device_id 0 --size base --mask_strategy_random none --mask_strategy fre --file_load_path ./weights/wifo_base --few_ratio 0.0 --t_patch_size 4 --patch_size 4 --batch_size 128 --pos_emb SinCos_3D


cd /home/blessedg/Pathformer/WiFo/src
python finetune_beam.py \
  --checkpoint ./weights/wifo_base \
  --all-scenarios \
  --epochs 20 \
  --batch-size 128 \
  --size base
```
## Citation
If you find this repo helpful, please cite our paper.
```latex
@article{liu2025wifo,
  author  = {Liu, Boxun and Gao, Shijian and Liu, Xuanyu and Cheng, Xiang and Yang, Liuqing},
  title   = {{WiFo: Wireless Foundation Model for Channel Prediction}},
  journal = {Sci. China Inf. Sci.},
  volume  = {68},
  pages   = {162302},
  year    = {2025},
  month   = {May},
  doi     = {10.1007/s11432-025-4349-0}


  conda run -n pathformer python /home/blessedg/Pathformer/WiFo/src/main.py \
  --dataset city_47_chicago_3p5 \
  --do_finetune true \
  --file_load_path ./weights/wifo_base  \
  --data_dir /home/blessedg/Pathformer/WiFo/dataset/blessed_task_user_loc \
  --freeze_wifo_backbone true \
  --train_decoder_pred true \
  --adapter_mode none \
  --input_noise_snr_db -1 \
  --finetune_epochs 20 \
  --device_id 0 \
  --note decoderpred_only \
  --t_patch_size 4 \
  --patch_size 4 \
  --pos_emb SinCos_3D 

python /home/blessedg/Pathformer/WiFo/src/precompute_pathformer_rollout_features.py \
  --scenario city_47_chicago_3p5 \
  --data-dir /home/blessedg/Pathformer/WiFo/dataset/blessed_task_user_loc

python /home/blessedg/Pathformer/WiFo/src/main.py \
  --dataset city_47_chicago_3p5 \
  --do_finetune true \
  --file_load_path ./weights/wifo_base  \
  --data_dir /home/blessedg/Pathformer/WiFo/dataset/blessed_task_user_loc \
  --freeze_wifo_backbone true \
  --train_decoder_pred true \
  --adapter_mode pathformer \
  --input_noise_snr_db -1 \
  --finetune_epochs 20 \
  --device_id 0 \
  --note decoderpred_only \
  --t_patch_size 4 \
  --patch_size 4 \
  --pos_emb SinCos_3D 

python /home/blessedg/Pathformer/WiFo/src/main.py \
  --dataset city_47_chicago_3p5 \
  --do_finetune true \
  --file_load_path ./weights/wifo_base  \
  --data_dir /home/blessedg/Pathformer/WiFo/dataset/blessed_task_user_loc \
  --freeze_wifo_backbone true \
  --train_decoder_pred true \
  --adapter_mode pathformer_tokens \
  --input_noise_snr_db -1 \
  --finetune_epochs 20 \
  --device_id 0 \
  --note decoderpred_only \
  --t_patch_size 4 \
  --patch_size 4 \
  --pos_emb SinCos_3D 


}
```


