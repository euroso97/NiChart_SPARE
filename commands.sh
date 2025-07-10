# Trainer

# CL
NiChart_SPARE -a trainer \
			  -t CL \
			  -i /home/kylebaik/Packages/NiChart_SPARE/Data/SPARE-AD-Harmonized-new.csv \
			  -mt SVM \
			  -sk linear \
			  -ht True \
			  -tw True \
			  -cf 5 \
			  -mo /home/kylebaik/Packages/NiChart_SPARE/Models/SPARE-AD-new.joblib \
			  -kv MRID \
			  -tc disease \
			  -ic Study,SITE,Sex \
			  -cb False \
			  -v 1

NiChart_SPARE -a trainer \
			  -t CL \
			  -i /home/kylebaik/Packages/NiChart_SPARE/Data/SPARE-AD-Harmonized-new.csv \
			  -mt SVM \
			  -sk linear_fast \
			  -ht True \
			  -tw True \
			  -cf 5 \
			  -mo /home/kylebaik/Packages/NiChart_SPARE/Models/SPARE-AD-new.joblib \
			  -kv MRID \
			  -tc disease \
			  -ic Study,SITE,Sex \
			  -cb False \
			  -v 1

NiChart_SPARE -a inference \
			  -t CL \
			  -i /home/kylebaik/Packages/NiChart_SPARE/Data/SPARE-AD-Harmonized-new.csv \
			  -m /home/kylebaik/Packages/NiChart_SPARE/Models/SPARE-AD-new.joblib \
			  -o /home/kylebaik/Packages/NiChart_SPARE/Data/Output_SPARE-AD-Harmonized-new.csv \
			  -kv MRID

# RG
NiChart_SPARE -a trainer \
			  -t RG \
			  -i /home/kylebaik/Packages/NiChart_SPARE/Data/SPARE-CONTROLNOOVERLAP-Harmonized-new.csv \
			  -mt SVM \
			  -sk linear \
			  -ht False \
			  -tw True \
			  -cf 5 \
			  -mo /home/kylebaik/Packages/NiChart_SPARE/Models/SPARE-BA-new.joblib \
			  -kv MRID \
			  -tc Age \
			  -ic Study,SITE,Sex \
			  -v 1
			  

NiChart_SPARE -a inference \
			  -t RG \
			  -i /home/kylebaik/Packages/NiChart_SPARE/Data/SPARE-CONTROLNOOVERLAP-Harmonized-new.csv \
			  -m /home/kylebaik/Packages/NiChart_SPARE/Models/SPARE-BA-new.joblib \
			  -o /home/kylebaik/Packages/NiChart_SPARE/Data/Output_SPARE-BA-Harmonized-new.csv \
			  -kv MRID

for disease in AD BMI DIABETES HYPERTENSION SMOKING; do
	NiChart_SPARE -a inference \
				-t RG \
				-i /home/kylebaik/Packages/NiChart_SPARE/Data/SPARE-${disease}-Harmonized-new.csv \
				-m /home/kylebaik/Packages/NiChart_SPARE/Models/SPARE-BA-new.joblib \
				-o /home/kylebaik/Packages/NiChart_SPARE/Output/Output_SPARE-BA-Harmonized-Diease_${disease}.csv \
				-kv MRID

# NiChart_SPARE -a trainer \
# 			  -t BA \
# 			  -i /home/kylebaik/Packages/NiChart_SPARE/Data/SPARE-BA-Harmonized-UKBIOBANK.csv \
# 			  -m /home/kylebaik/Packages/NiChart_SPARE/Models/SPARE-BA-new-ukbiobank.joblib \
# 			  -o /home/kylebaik/Packages/NiChart_SPARE/Data/Output_SPARE-BA-Harmonized-UKBIOBANK.csv
