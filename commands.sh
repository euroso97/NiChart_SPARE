# Trainer

NiChart_SPARE -a trainer \
			  -t CL \
			  -i /home/kylebaik/Packages/NiChart_SPARE/Data/SPARE-AD-Harmonized.csv \
			  -mt SVM \
			  -sk linear \
			  -ht True \
			  -tw True \
			  -cf 5 \
			  -mo /home/kylebaik/Packages/NiChart_SPARE/Models/SPARE-AD-new.joblib \
			  -kv MRID \
			  -tc DX \
			  -ic Study,SITE \
			  -cb True \
			  -v 1

NiChart_SPARE -a trainer \
			  -t CL \
			  -i /home/kylebaik/Packages/NiChart_SPARE/Data/SPARE-AD-Harmonized.csv \
			  -mt SVM \
			  -sk rbf \
			  -ht False \
			  -tw True \
			  -cf 5 \
			  -mo /home/kylebaik/Packages/NiChart_SPARE/Models/SPARE-AD-new.joblib \
			  -kv MRID \
			  -tc DX \
			  -ic Study,SITE \
			  -cb True \
			  -v 1
			  

# Inference 

NiChart_SPARE -a inference \
			  -t CL \
			  -i /home/kylebaik/Packages/NiChart_SPARE/Data/SPARE-AD-Harmonized.csv \
			  -m /home/kylebaik/Packages/NiChart_SPARE/Models/SPARE-AD-new.joblib \
			  -o /home/kylebaik/Packages/NiChart_SPARE/Data/Output_SPARE-AD-Harmonized.csv

NiChart_SPARE -a trainer \
			  -t BA \
			  -i /home/kylebaik/Packages/NiChart_SPARE/Data/SPARE-BA-Harmonized-UKBIOBANK.csv \
			  -m /home/kylebaik/Packages/NiChart_SPARE/Models/SPARE-BA-new-ukbiobank.joblib \
			  -o /home/kylebaik/Packages/NiChart_SPARE/Data/Output_SPARE-BA-Harmonized-UKBIOBANK.csv
