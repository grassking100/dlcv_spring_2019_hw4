# TODO: create shell script for Problem 1
input=$1
answer=$2
output=$3
wget https://www.dropbox.com/s/asjr8dbjo7itvwm/answer_1_classifier.pth?dl=1
model_path=answer_1_classifier.pth?dl=1
python3 p1_predict.py $input $answer $output $model_path