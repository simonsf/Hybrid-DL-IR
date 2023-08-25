## Preparation

- Input Data: Low dose CT image or PET image. 
- Model parameters files: refer to models directory in this repo.

-------------------------------
## Run
```
python apply_CT -i path_to_input_image -m directory_to_model -o directory_to_output_image -g gpu_id
```
and
```
python apply_PET -i path_to_input_image -m directory_to_model -o directory_to_output_image -g gpu_id
```

-------------------------------
## Requirements
numpy, SimpleITK, easydict, torch
Note that the models folder was not uploaded here since the size exceeds 25MB of Github limit. Please refer to the Mendeley Data website (https://www.doi.org/10.17632/j7khwb3z3r) where we uploaded the codes too.
