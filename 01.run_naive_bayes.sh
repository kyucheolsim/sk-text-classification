# written by kylesim

CLF="mnb"
#CLF="cnb"
#CLF="gnb"
MAX_FEATURES=8000
MIN_DF=20
MAX_DF=0.7
ALPHA=0.5

BIN_TRAIN="train_naive_bayes.py"
BIN_PREDICT="predict_classifier.py"
DATA_PATH="./data/txt_sentoken"
GS_MODEL_PATH="./model/model_naive_bayes_gs.bin"
MAIN_MODEL_PATH="./model/model_naive_bayes_${CLF}.bin"

run_train()
{
	set -x
	if [ "$1" == "gs" ]; then
		python3 ${BIN_TRAIN} --classifier=${CLF} --data_path=${DATA_PATH} --model_path=${GS_MODEL_PATH} --grid_search
	elif [ "$1" == "main" ]; then
		python3 ${BIN_TRAIN} --classifier=${CLF} --data_path=${DATA_PATH} --model_path=${MAIN_MODEL_PATH} --max_features=${MAX_FEATURES} --min_df=${MIN_DF} --max_df=${MAX_DF} --alpha=${ALPHA}
	fi
}

run_predict()
{
	MODEL_PATH=${GS_MODEL_PATH}
	#MODEL_PATH=${MAIN_MODEL_PATH}

	set -x
	if [ "$1" == "batch" ]; then
		python3 ${BIN_PREDICT} --action=batch --data_path=${DATA_PATH} --model_path=${MODEL_PATH} --labeled
	elif [ "$1" == "accuracy" ]; then
		set -x
		python3 ${BIN_PREDICT} --action=accuracy --data_path=${DATA_PATH} --model_path=${MODEL_PATH} --labeled --debug
	fi
}

case $1 in
	'train')
		shift
		run_train $1
	;;
	'predict')
		shift
		run_predict $1
esac
