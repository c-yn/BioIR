

## Installation and Data Preparation

See [INSTALL.md](INSTALL.md) for the installation of dependencies and dataset preperation.

## Training

After preparing the training data in ```data/``` directory, use 
```
python train_eval.py
```
to start the training of the model. Use ```de_type``` to choose the combination of degradation types to train on. By default it is set to all the 5 degradation tasks (denoising ($\sigma$=15/25/50), deraining, dehazing, deblurring, enhancement).

Example: If we only want to train the model in the three-task setting:
```
python train.py --de_type derain dehaze denoise_15 denoise_25 denoise_50
```

## Testing

After preparing the testing data in ```test/``` directory, place the mode checkpoint file in the ```ckpt``` directory. To perform the evaluation, use
```
cd All_in_One
python test.py --mode {n} 
```
```n``` is a number that can be used to set the tasks to be evaluated on: 
0 for denoising, 1 for deraining, 2 for dehazing, 3 for deblurring, 4 for enhancement, 5 for three-task setting, and 6 for five-task setting.

Example: After placing the pretrained model in ```ckpt/```, run:

```
python test.py --mode 6 --ckpt_name BioIR5D.ckpt
```
to test on all the degradation types
## Demo


To obtain visual results from the model, ``demo.py`` can be used. After placing the pre-trained models of three-task or five-task settings in ``ckpt`` directory, run:
```
cd All_in_One
python demo.py --ckpt_name BioIR5D.ckpt --test_path {path_to_degraded_images} --output_path {save_images_here} 
```

Example usage to run inference on a directory of images:
```
cd All_in_One
python demo.py --ckpt_name BioIR5D.ckpt  --test_path './demo/degraded/' --output_path './demo/restored/'
```

Example usage to run inference on an image directly:

```
cd All_in_One
python demo.py --ckpt_name BioIR5D.ckpt  --test_path './demo/degraded/1.jpg' --output_path './demo/degraded/'
```

To use tiling option while running ``demo.py`` set ``--tile`` option to ``True``. The Tile size and Tile overlap parameters can be adjusted using ``--tile_size`` and ``--tile_overlap`` options respectively.
