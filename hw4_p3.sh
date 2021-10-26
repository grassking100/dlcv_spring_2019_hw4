# TODO: create shell script for Problem 3
input=$1
output=$2
wget https://www.dropbox.com/s/m5y4zeurges2785/answer_3_56.37_classifier.pth?dl=1
model_path=answer_3_56.37_classifier.pth?dl=1
python3 p3_predict.py $input $output $model_path