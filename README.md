# FACE RECOGNITION BASED ATTENDANCE SYSTEM

## INCLUDED FILE AND DIRECTORIES 
- app.py : the main user-interface for the project
- model.py : it contains the architecture of our model and hyperparameters
- add_students.py : the code for adding new students' data to the attendance system
- attendance.py : the code for marking attendance of students
- train_model.py : code for retraining the model whenever a new student is added
- student_list.json : json file with the data of the students
- attendance.csv : csv file where all the attendance is marked
- students : directory where student images are stored (not included in the git repo)
- output : model weights are stored here after training (not included in the git repo)

## DEPENDENCIES
- opencv-python
- pytorch
- facenet_pytorch
- torchvision
- transformers
- sklearn
- numpy

## HOW TO USE
- download the repo, add an (empty) folder named `students` at the project location
- add new students data using `app.py`
- this will save their images in the `students` folder, and their data in the `student_list.json` file
- after adding new students, train the model using `train_model.py`
- mark and view attendance using `app.py`

NOTE: if the model does not give optimal performance on your image dataset, you might want to
- introduce variation (lighting conditions and capture angles)
- tune the hyperparameters in `model.py`
- `train_model.py` returns the graphs of accuracies and f1-scores to help in fine tuning by analyzing the models performance
