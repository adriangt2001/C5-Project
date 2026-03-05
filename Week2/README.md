## Main
The **main.py** file is the entrypoint for every task (similar to C6). Add every argument needed there. If you want more modularity change it as you want. It already supports config files in .yaml format, so you can automatize experiments with shorter commands from the beginning!

## Utilities
**Prompt class**: inside **utils/prompts.py** there is a class for managing prompts (can access it as a normal array, use its method to add more prompts, can also get full list of prompts if you want. They can be saved into a .txt or .npy file, depending on the prompt list mode.) Look into the code and you will understand it easier (I hope) :)

**Dataset class**: inside **utils/kitti_dataset.py** there is a class for loading the KITTI-MOTS dataset as a torch dataset. I didn't test augmentations so be careful. In "test" mode no masks will be loaded.

**Interactive**: inside **utils/interactive.py** there is a function to select a point by clicking in an image. It won't work in remote but can be used in local first and then store the points and upload them to the remote.

**Visualizations**: inside **utils/visualizations.py** there is a function to create a video. Won't be useful for the final slides, but it may help you to check the whole dataset faster than one image at a time (even with masks already segmented).

If you need any other functionality add it to the corresponding file if it can be reused for different tasks (you can add another file if it doesn't fit in any of the already existing ones). Otherwise, just add it to the specific task.

## Folder structure
The results, datasets and checkpoints folders are in the gitignore. If you think another folder will be useful to keep other things organized go ahead and create it (for example for bash scripts that may help you run several experiments at once).

## Idk what to name this section
If you do any hyperparameter search (or any other search idk) look at optuna. I saw other groups using it and it seems to work ok, although I haven't used it. I saw it has integration with huggingface when training or something like that, but for what I saw in the slides Carles showed us, the training loop in finetuning was done using a normal pytorch loop. You decide ^^

## Nothing else
If you don't like this or want a different structure feel free to change anything or even nuke the whole Week2 folder c:
