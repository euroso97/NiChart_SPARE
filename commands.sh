# Trainer

NiChart_SPARE -a trainer \
			  -t AD \
			  -i /home/kylebaik/Packages/NiChart_SPARE/Data/SPARE-AD-Harmonized.csv \
			  -tc DX \
			  -kv MRID \
			  -iv Study,SITE \
			  -cb True \
			  -mo /home/kylebaik/Packages/NiChart_SPARE/Models/SPARE-AD-new.joblib \
			  -mt SVM \
			  -sk linear

NiChart_SPARE -a trainer \
			  -t BA \
			  -i /home/kylebaik/Packages/NiChart_SPARE/Data/SPARE-BA-Harmonized-UKBIOBANK.csv \
			  -tc Age \
			  -kv MRID \
			  -iv Study,SITE \
			  -cb True \
			  -mo /home/kylebaik/Packages/NiChart_SPARE/Models/SPARE-BA-new-ukbiobank.joblib \
			  -mt SVM \
			  -sk linear

# Inference 

NiChart_SPARE -a inference \
			  -t AD \
			  -i /home/kylebaik/Packages/NiChart_SPARE/Data/SPARE-AD-Harmonized.csv \
			  -m /home/kylebaik/Packages/NiChart_SPARE/Models/SPARE-AD-new.joblib \
			  -o /home/kylebaik/Packages/NiChart_SPARE/Data/Output_SPARE-AD-Harmonized.csv

NiChart_SPARE -a trainer \
			  -t BA \
			  -i /home/kylebaik/Packages/NiChart_SPARE/Data/SPARE-BA-Harmonized-UKBIOBANK.csv \
			  -m /home/kylebaik/Packages/NiChart_SPARE/Models/SPARE-BA-new-ukbiobank.joblib \
			  -o /home/kylebaik/Packages/NiChart_SPARE/Data/Output_SPARE-BA-Harmonized-UKBIOBANK.csv
