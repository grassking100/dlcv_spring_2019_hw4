# TODO: create shell script for Problem 2
input=$1
answer=$2
output=$3
wget https://www.dropbox.com/s/9vht742se2ygjej/answer_2_classifier_46.55.pth?dl=1
model_path=answer_2_classifier_46.55.pth?dl=1
python3 p2_predict.py $input $answer $output $model_path