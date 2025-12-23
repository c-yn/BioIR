



<h1 align="center">Bio-Inspired Image Restoration
</h1>

:star: If BioIR is helpful for you, please help star this repo. Thanks! :hugs:

## <a name="update"></a>:new: News

- **2025.12.23**: :fire: This repo is released
🎄 Merry Christmas! and Happy New Year!

## <a name="setup"></a> ⚙️ Setup

See [INSTALL.md](INSTALL.md) for the installation of dependencies required to run BioIR.


## <a name="training"></a> :wrench: Training and Evaluation

Training and Testing instructions for single-degradation&composite degradation and all-in-one tasks are provided in Single_Composite and All_in_One directories, respectively.

|Tasks|Instructions|Visual Results|Pre-trained Models|Datasets|
|---|---|---|---|---|
|Single and composite degradation|<h6 align="center">[Link](./Single_Composite/README.md)</h6>|<h6 align="center">[百度网盘](https://pan.baidu.com/s/18EIFlLx-xSQRIoLc62Qt6A?pwd=x65n)</h6>|<h6 align="center">[Google Drive](https://drive.google.com/drive/folders/1VrFxqox3fewPUmP-i0a9rJw3qCmT1Vnp?usp=sharing), [百度网盘](https://pan.baidu.com/s/1AEieYLl5i-afkr-bF47a_g?pwd=ja58)</h6>|<h6 align="center">Desnowing ([Google Drive](https://drive.google.com/drive/folders/1-KVJEk25jV3Ds4AG0QJwB8VKYx9GgNGV?usp=sharing), [百度网盘](https://pan.baidu.com/s/1AhijfTRP8ECdmR_UiqoO_Q?pwd=a5pz)). Dehazing ([Google Drive](https://drive.google.com/drive/folders/1-Jc7IujbGtfVLuUISZUcRWD1L5prsNqT?usp=sharing), [百度网盘](https://pan.baidu.com/s/1yTDmn6SfGtyQJE6-7V8WlA?pwd=sxdb)). CDD([百度网盘](https://pan.baidu.com/s/1ywdstQ8iN0Solb2Srx-PKw?pwd=sbqi)). LOLBlur ([百度网盘](https://pan.baidu.com/s/1LFrpLu-8oDOsMQ972ixIWw?pwd=dfte))|
|All-in-one|<h6 align="center">[Link](./All_in_One/README.md)</h6>|<h6 align="center">[Google Drive](https://drive.google.com/drive/folders/13vVScuqQXbJ6J0Xg4fXy_cEqk3Nxk6G_?usp=sharing), [百度网盘](https://pan.baidu.com/s/1C4zIRHJ0CGAq-Se-cSdkcg?pwd=82k4)</h6>|<h6 align="center">[Google Drive](https://drive.google.com/drive/folders/1tBv-4ixucMUxPCnVr3ZHf9EHlUnVM1vD?usp=sharing), [百度网盘](https://pan.baidu.com/s/1vGVAlMbY-GupxWykf9rj4g?pwd=2172)</h6>|<h6 align="center">[百度网盘](https://pan.baidu.com/s/1Ld3UGn_q9cBjrTwfjYlA9w?pwd=vbjv)|

## <a name="inference"></a> 💫 Demo
(For all-in-one demo, please refer to [demo.py](All_in_One/demo.py).)
To test the pre-trained BioIR models on your own images, you can download the models, place them in ```pretrain_model```.

Example: use the model pretrained on CDD on your own images:
```
python demo.py --input_dir './demo/degraded/' --result_dir './demo/restored/' --dataset CDD
```

## :notebook: Citation

Please cite us if our work is useful for your research.

```
@inproceedings{bioir,
title={Bio-Inspired Image Restoration},
author={Yuning Cui and Wenqi Ren and Alois Knoll},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025}
}

```

## :envelope: Contact

Should you have any question, please contact yuning.cui@in.tum.de
