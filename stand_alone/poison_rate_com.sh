conda init bash
conda activate ctr
cd ~/Public/ctr/

group_name="clear"
python ./stand_alone/poison_rate_com.py ${group_name}

group_name="random"
python ./stand_alone/poison_rate_com.py ${group_name}

group_name="model_dependent"
python ./stand_alone/poison_rate_com.py ${group_name}
