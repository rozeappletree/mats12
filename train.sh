echo "======================================================="
echo "Defense only, tested on hard"
echo "======================================================="

python src/train_reading_probe.py --dataset_dirs datasets_defense_484 --output_dir "probe_checkpoints.withDefense484Only" --run_name reading_probe
python src/train_control_probe.py --dataset_dirs datasets_defense_484 --output_dir "probe_checkpoints.withDefense484Only" --run_name control_probe

python src/test_reading_probe.py --checkpoint_dir "probe_checkpoints.withDefense484Only/reading_probe" --test_dirs datasets_hard_333

python src/test_control_probe.py --checkpoint_dir "probe_checkpoints.withDefense484Only/control_probe" --test_dirs datasets_hard_333


echo "======================================================="
echo "Defence and Regular Mixed, test on hard"
echo "======================================================="

# BIG LIMIT: Sequential Nature of Conversation because of Prompt Optimisation (can be corrected, need time)
# TEMP SOLN: K-FOLD SPLIT
python src/train_reading_probe.py --dataset_dirs datasets_defense_484 datasets_regular_gullibility_170 --output_dir "probe_checkpoints.withDefense484andRegular170" --run_name reading_probe
python src/train_control_probe.py --dataset_dirs datasets_defense_484 datasets_regular_gullibility_170 --output_dir "probe_checkpoints.withDefense484andRegular170" --run_name control_probe

python src/test_reading_probe.py --checkpoint_dir "probe_checkpoints.withDefense484andRegular170/reading_probe" --test_dirs datasets_hard_333

python src/test_control_probe.py --checkpoint_dir "probe_checkpoints.withDefense484andRegular170/control_probe" --test_dirs datasets_hard_333


echo "======================================================="
echo "Defence and Regualar and JustHard, tested on ardulent"
echo "======================================================="



# BIG LIMIT: Sequential Nature of Conversation because of Prompt Optimisation (can be corrected, need time)
# TEMP SOLN: K-FOLD SPLIT
python src/train_reading_probe.py --dataset_dirs datasets_defense_484 datasets_regular_gullibility_170 datasets_justhard_267 --output_dir "probe_checkpoints.withDefense484Regular170JustHard267" --run_name reading_probe
python src/train_control_probe.py --dataset_dirs datasets_defense_484 datasets_regular_gullibility_170 datasets_justhard_267 --output_dir "probe_checkpoints.withDefense484Regular170JustHard267" --run_name control_probe

python src/test_reading_probe.py --checkpoint_dir "probe_checkpoints.withDefense484Regular170JustHard267/reading_probe" --test_dirs datasets_ardulous_66

python src/test_control_probe.py --checkpoint_dir "probe_checkpoints.withDefense484Regular170JustHard267/control_probe" --test_dirs datasets_ardulous_66

