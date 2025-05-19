python src/schema_generation.py --model_name "llama" --method "text" --prompt_type "direct"
python src/schema_evaluation.py --model_name "llama" --method "text" --prompt_type "direct"

python src/value_identification.py --model_name "llama" --method "symbolic" --dataset "bird"
python src/value_identification.py --model_name "llama" --method "llm" --dataset "bird"

python src/value_population.py --model_name "llama" --method "TS" --dataset "bird"
python src/value_population.py --model_name "llama" --method "TST" --dataset "bird"
python src/value_population.py --model_name "llama" --method "TST-L" --dataset "bird"

python src/database_generation.py --method "TS" --model_name "llama" --dataset "bird"
python src/database_generation.py --method "TST" --model_name "llama" --dataset "bird"
python src/database_generation.py --method "TST-L" --model_name "llama" --dataset "bird"

python src/database_evaluation.py --method "TS" --model_name "llama" --dataset "bird"
python src/database_evaluation.py --method "TST" --model_name "llama" --dataset "bird"
python src/database_evaluation.py --method "TST-L" --model_name "llama" --dataset "bird"

python helpers/ensemble.py
