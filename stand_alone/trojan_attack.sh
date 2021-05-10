###### GROUP NAME ######
group_name="random" # random | model_dependent

# To run background:
# nohup bash stand_alone/trojan_attack.sh > trojan_attack.log 2>&1 &

### 1. Poison Rate Comparison ###
# trigger_size=0.2

# poison_rate=0.001
# python ./stand_alone/trojan_attack.py ${group_name} ${poison_rate} ${trigger_size}

# poison_rate=0.01
# python ./stand_alone/trojan_attack.py ${group_name} ${poison_rate} ${trigger_size}

# poison_rate=0.05
# python ./stand_alone/trojan_attack.py ${group_name} ${poison_rate} ${trigger_size}

# poison_rate=0.08
# python ./stand_alone/trojan_attack.py ${group_name} ${poison_rate} ${trigger_size}

# poison_rate=0.1
# python ./stand_alone/trojan_attack.py ${group_name} ${poison_rate} ${trigger_size}

### 2. Trigger Size Comparison ###
poison_rate=0.1

trigger_size=0.03
python ./stand_alone/trojan_attack.py ${group_name} ${poison_rate} ${trigger_size}

# trigger_size=0.1
# python ./stand_alone/trojan_attack.py ${group_name} ${poison_rate} ${trigger_size}

# trigger_size=0.15
# python ./stand_alone/trojan_attack.py ${group_name} ${poison_rate} ${trigger_size}

# trigger_size=0.2
# python ./stand_alone/trojan_attack.py ${group_name} ${poison_rate} ${trigger_size}


### 3. Trigger Field Comparison ###




