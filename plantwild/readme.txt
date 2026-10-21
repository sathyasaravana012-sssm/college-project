===========================================
PlantWild Dataset
===========================================


=========================
IMAGES AND CLASS LABELS:
=========================
Images are contained in the directory images/, with 89 subdirectories.


------- List of class names (classes.txt) ------
The list of class names is contained in the file classes.txt, with each line corresponding to one class:

<class_id> <class_name>



------- Train/val/test split (split.txt) ------
The suggested train/val/test split is contained in the file split.txt, with each line corresponding to one image:

<image_path>=<class_id>=<mode>

where <image_path> corresponds to the relative paths of images, <class_id> correspond the IDs in classes.txt, and a value of 0, 1, 2 for <mode> denotes that the file is in the test, training or validation set, respectively.



------- Text prompts for each class (plantwild_promptps.json) ------
The GPT-generated descriptive prompts, taken as the input of CLIP's textual encoder. For each class there are 50 prompts.


------- ULRs of image sources (url_record.json) ------
The record file for our dataset includes the path of each image and its download link.

------------------------------------------------------