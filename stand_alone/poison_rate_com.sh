# conda init bash
# conda activate ctr
# cd /home/zhimin/Trojan_Attack_Against_DeepFM_FL/

# group_name="clear"
# python ./stand_alone/poison_rate_com.py ${group_name}

# group_name="random"
# python ./stand_alone/poison_rate_com.py ${group_name}

# group_name="model_dependent"
# python ./stand_alone/poison_rate_com.py ${group_name}

# RANDOM GROUP
group_name="random"
python ./stand_alone/trojan_attack.py ${group_name}


