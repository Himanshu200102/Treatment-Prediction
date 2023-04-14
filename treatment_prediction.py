from tkinter import *
from tkinter import messagebox
import numpy as np
import pandas as pd

l1=['itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering','chills','joint_pain',
    'stomach_pain','acidity','ulcers_on_tongue','muscle_wasting','vomiting','burning_micturition','spotting_ urination','fatigue',
    'weight_gain','anxiety','cold_hands_and_feets','mood_swings','weight_loss','restlessness','lethargy','patches_in_throat',
    'irregular_sugar_level','cough','high_fever','sunken_eyes','breathlessness','sweating','dehydration','indigestion',
    'headache','yellowish_skin','dark_urine','nausea','loss_of_appetite','pain_behind_the_eyes','back_pain','constipation',
    'abdominal_pain','diarrhoea','mild_fever','yellow_urine','yellowing_of_eyes','acute_liver_failure','fluid_overload',
    'swelling_of_stomach','swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
    'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs','fast_heart_rate',
    'pain_during_bowel_movements','pain_in_anal_region','bloody_stool','irritation_in_anus','neck_pain','dizziness','cramps',
    'bruising','obesity','swollen_legs','swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
    'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips','slurred_speech','knee_pain','hip_joint_pain',
    'muscle_weakness','stiff_neck','swelling_joints','movement_stiffness','spinning_movements','loss_of_balance','unsteadiness','weakness_of_one_body_side',
    'loss_of_smell','bladder_discomfort','foul_smell_of urine','continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
    'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain','abnormal_menstruation','dischromic _patches',
    'watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum','rusty_sputum','lack_of_concentration','visual_disturbances',
    'receiving_blood_transfusion','receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen','history_of_alcohol_consumption',
    'fluid_overload','blood_in_sputum','prominent_veins_on_calf','palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
    'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose','yellow_crust_ooze']

disease=['(vertigo) Paroymsal  Positional Vertigo T1',
'(vertigo) Paroymsal  Positional Vertigo T2',
'(vertigo) Paroymsal  Positional Vertigo T3',
'(vertigo) Paroymsal  Positional Vertigo T4',
'(vertigo) Paroymsal  Positional Vertigo T5',
'(vertigo) Paroymsal  Positional Vertigo T6',
'(vertigo) Paroymsal  Positional Vertigo T7',
'(vertigo) Paroymsal  Positional Vertigo T8',
'(vertigo) Paroymsal  Positional Vertigo T9',
'(vertigo) Paroymsal  Positional Vertigo T10',
'Acne T11',
'Acne T12',
'Acne T13',
'Acne T14',
'Acne T15',
'Acne T16',
'Acne T17',
'Acne T18',
'Acne T19',
'Acne T20',
'AIDS T21',
'AIDS T22',
'AIDS T23',
'AIDS T24',
'AIDS T25',
'AIDS T26',
'AIDS T27',
'AIDS T28',
'AIDS T29',
'AIDS T30',
'Alcoholic hepatitis T31',
'Alcoholic hepatitis T32',
'Alcoholic hepatitis T33',
'Alcoholic hepatitis T34',
'Alcoholic hepatitis T35',
'Alcoholic hepatitis T36',
'Alcoholic hepatitis T37',
'Alcoholic hepatitis T38',
'Alcoholic hepatitis T39',
'Alcoholic hepatitis T40',
'Allergy T41',
'Allergy T42',
'Allergy T43',
'Allergy T44',
'Allergy T45',
'Allergy T46',
'Allergy T47',
'Allergy T48',
'Allergy T49',
'Allergy T50',
'Arthritis T51',
'Arthritis T52',
'Arthritis T53',
'Arthritis T54',
'Arthritis T55',
'Arthritis T56',
'Arthritis T57',
'Arthritis T58',
'Arthritis T59',
'Arthritis T60',
'Bronchial Asthma T61',
'Bronchial Asthma T62',
'Bronchial Asthma T63',
'Bronchial Asthma T64',
'Bronchial Asthma T65',
'Bronchial Asthma T66',
'Bronchial Asthma T67',
'Bronchial Asthma T68',
'Bronchial Asthma T69',
'Bronchial Asthma T70',
'Cervical spondylosis T71',
'Cervical spondylosis T72',
'Cervical spondylosis T73',
'Cervical spondylosis T74',
'Cervical spondylosis T75',
'Cervical spondylosis T76',
'Cervical spondylosis T77',
'Cervical spondylosis T78',
'Cervical spondylosis T79',
'Cervical spondylosis T80',
'Chicken pox T81',
'Chicken pox T82',
'Chicken pox T83',
'Chicken pox T84',
'Chicken pox T85',
'Chicken pox T86',
'Chicken pox T87',
'Chicken pox T88',
'Chicken pox T89',
'Chicken pox T90',
'Chronic cholestasis T91',
'Chronic cholestasis T92',
'Chronic cholestasis T93',
'Chronic cholestasis T94',
'Chronic cholestasis T95',
'Chronic cholestasis T96',
'Chronic cholestasis T97',
'Chronic cholestasis T98',
'Chronic cholestasis T99',
'Chronic cholestasis T100',
'Common Cold T101',
'Common Cold T102',
'Common Cold T103',
'Common Cold T104',
'Common Cold T105',
'Common Cold T106',
'Common Cold T107',
'Common Cold T108',
'Common Cold T109',
'Common Cold T110',
'Dengue T111',
'Dengue T112',
'Dengue T113',
'Dengue T114',
'Dengue T115',
'Dengue T116',
'Dengue T117',
'Dengue T118',
'Dengue T119',
'Dengue T120',
'Diabetes  T1',
'Diabetes  T2',
'Diabetes  T3',
'Diabetes  T4',
'Diabetes  T5',
'Diabetes  T6',
'Diabetes  T7',
'Diabetes  T8',
'Diabetes  T9',
'Diabetes  T10',
'Dimorphic hemmorhoids(piles) T11',
'Dimorphic hemmorhoids(piles) T12',
'Dimorphic hemmorhoids(piles) T13',
'Dimorphic hemmorhoids(piles) T14',
'Dimorphic hemmorhoids(piles) T15',
'Dimorphic hemmorhoids(piles) T16',
'Dimorphic hemmorhoids(piles) T17',
'Dimorphic hemmorhoids(piles) T18',
'Dimorphic hemmorhoids(piles) T19',
'Dimorphic hemmorhoids(piles) T20',
'Drug Reaction T21',
'Drug Reaction T22',
'Drug Reaction T23',
'Drug Reaction T24',
'Drug Reaction T25',
'Drug Reaction T26',
'Drug Reaction T27',
'Drug Reaction T28',
'Drug Reaction T29',
'Drug Reaction T30',
'Fungal infection T31',
'Fungal infection T32',
'Fungal infection T33',
'Fungal infection T34',
'Fungal infection T35',
'Fungal infection T36',
'Fungal infection T37',
'Fungal infection T38',
'Fungal infection T39',
'Fungal infection T40',
'Gastroenteritis T41',
'Gastroenteritis T42',
'Gastroenteritis T43',
'Gastroenteritis T44',
'Gastroenteritis T45',
'Gastroenteritis T46',
'Gastroenteritis T47',
'Gastroenteritis T48',
'Gastroenteritis T49',
'Gastroenteritis T50',
'GERD T51',
'GERD T52',
'GERD T53',
'GERD T54',
'GERD T55',
'GERD T56',
'GERD T57',
'GERD T58',
'GERD T59',
'GERD T60',
'Heart attack T61',
'Heart attack T62',
'Heart attack T63',
'Heart attack T64',
'Heart attack T65',
'Heart attack T66',
'Heart attack T67',
'Heart attack T68',
'Heart attack T69',
'Heart attack T70',
'hepatitis A T71',
'hepatitis A T72',
'hepatitis A T73',
'hepatitis A T74',
'hepatitis A T75',
'hepatitis A T76',
'hepatitis A T77',
'hepatitis A T78',
'hepatitis A T79',
'hepatitis A T80',
'Hepatitis B T81',
'Hepatitis B T82',
'Hepatitis B T83',
'Hepatitis B T84',
'Hepatitis B T85',
'Hepatitis B T86',
'Hepatitis B T87',
'Hepatitis B T88',
'Hepatitis B T89',
'Hepatitis B T90',
'Hepatitis C T91',
'Hepatitis C T92',
'Hepatitis C T93',
'Hepatitis C T94',
'Hepatitis C T95',
'Hepatitis C T96',
'Hepatitis C T97',
'Hepatitis C T98',
'Hepatitis C T99',
'Hepatitis C T100',
'Hepatitis D T101',
'Hepatitis D T102',
'Hepatitis D T103',
'Hepatitis D T104',
'Hepatitis D T105',
'Hepatitis D T106',
'Hepatitis D T107',
'Hepatitis D T108',
'Hepatitis D T109',
'Hepatitis D T110',
'Hepatitis E T111',
'Hepatitis E T112',
'Hepatitis E T113',
'Hepatitis E T114',
'Hepatitis E T115',
'Hepatitis E T116',
'Hepatitis E T117',
'Hepatitis E T118',
'Hepatitis E T119',
'Hepatitis E T120',
'Hypertension  T1',
'Hypertension  T2',
'Hypertension  T3',
'Hypertension  T4',
'Hypertension  T5',
'Hypertension  T6',
'Hypertension  T7',
'Hypertension  T8',
'Hypertension  T9',
'Hypertension  T10',
'Hyperthyroidism T11',
'Hyperthyroidism T12',
'Hyperthyroidism T13',
'Hyperthyroidism T14',
'Hyperthyroidism T15',
'Hyperthyroidism T16',
'Hyperthyroidism T17',
'Hyperthyroidism T18',
'Hyperthyroidism T19',
'Hyperthyroidism T20',
'Hypoglycemia T21',
'Hypoglycemia T22',
'Hypoglycemia T23',
'Hypoglycemia T24',
'Hypoglycemia T25',
'Hypoglycemia T26',
'Hypoglycemia T27',
'Hypoglycemia T28',
'Hypoglycemia T29',
'Hypoglycemia T30',
'Hypothyroidism T31',
'Hypothyroidism T32',
'Hypothyroidism T33',
'Hypothyroidism T34',
'Hypothyroidism T35',
'Hypothyroidism T36',
'Hypothyroidism T37',
'Hypothyroidism T38',
'Hypothyroidism T39',
'Hypothyroidism T40',
'Impetigo T41',
'Impetigo T42',
'Impetigo T43',
'Impetigo T44',
'Impetigo T45',
'Impetigo T46',
'Impetigo T47',
'Impetigo T48',
'Impetigo T49',
'Impetigo T50',
'Jaundice T51',
'Jaundice T52',
'Jaundice T53',
'Jaundice T54',
'Jaundice T55',
'Jaundice T56',
'Jaundice T57',
'Jaundice T58',
'Jaundice T59',
'Jaundice T60',
'Malaria T61',
'Malaria T62',
'Malaria T63',
'Malaria T64',
'Malaria T65',
'Malaria T66',
'Malaria T67',
'Malaria T68',
'Malaria T69',
'Malaria T70',
'Migraine T71',
'Migraine T72',
'Migraine T73',
'Migraine T74',
'Migraine T75',
'Migraine T76',
'Migraine T77',
'Migraine T78',
'Migraine T79',
'Migraine T80',
'Osteoarthristis T81',
'Osteoarthristis T82',
'Osteoarthristis T83',
'Osteoarthristis T84',
'Osteoarthristis T85',
'Osteoarthristis T86',
'Osteoarthristis T87',
'Osteoarthristis T88',
'Osteoarthristis T89',
'Osteoarthristis T90',
'Paralysis (brain hemorrhage) T91',
'Paralysis (brain hemorrhage) T92',
'Paralysis (brain hemorrhage) T93',
'Paralysis (brain hemorrhage) T94',
'Paralysis (brain hemorrhage) T95',
'Paralysis (brain hemorrhage) T96',
'Paralysis (brain hemorrhage) T97',
'Paralysis (brain hemorrhage) T98',
'Paralysis (brain hemorrhage) T99',
'Paralysis (brain hemorrhage) T100',
'Peptic ulcer diseae T101',
'Peptic ulcer diseae T102',
'Peptic ulcer diseae T103',
'Peptic ulcer diseae T104',
'Peptic ulcer diseae T105',
'Peptic ulcer diseae T106',
'Peptic ulcer diseae T107',
'Peptic ulcer diseae T108',
'Peptic ulcer diseae T109',
'Peptic ulcer diseae T110',
'Pneumonia T111',
'Pneumonia T112',
'Pneumonia T113',
'Pneumonia T114',
'Pneumonia T115',
'Pneumonia T116',
'Pneumonia T117',
'Pneumonia T118',
'Pneumonia T119',
'Pneumonia T120',
'Psoriasis T1',
'Psoriasis T2',
'Psoriasis T3',
'Psoriasis T4',
'Psoriasis T5',
'Psoriasis T6',
'Psoriasis T7',
'Psoriasis T8',
'Psoriasis T9',
'Psoriasis T10',
'Tuberculosis T11',
'Tuberculosis T12',
'Tuberculosis T13',
'Tuberculosis T14',
'Tuberculosis T15',
'Tuberculosis T16',
'Tuberculosis T17',
'Tuberculosis T18',
'Tuberculosis T19',
'Tuberculosis T20',
'Typhoid T21',
'Typhoid T22',
'Typhoid T23',
'Typhoid T24',
'Typhoid T25',
'Typhoid T26',
'Typhoid T27',
'Typhoid T28',
'Typhoid T29',
'Typhoid T30',
'Urinary tract infection T31',
'Urinary tract infection T32',
'Urinary tract infection T33',
'Urinary tract infection T34',
'Urinary tract infection T35',
'Urinary tract infection T36',
'Urinary tract infection T37',
'Urinary tract infection T38',
'Urinary tract infection T39',
'Urinary tract infection T40',
'Varicose veins T41',
'Varicose veins T42',
'Varicose veins T43',
'Varicose veins T44',
'Varicose veins T45',
'Varicose veins T46',
'Varicose veins T47',
'Varicose veins T48',
'Varicose veins T49',
'Varicose veins T50']

l2=[]
for x in range(0,len(l1)):
    l2.append(0)

# TESTING DATA
tr=pd.read_csv("Testing.csv")
tr.replace({'prognosis':{'Fungal infection T':0,'Fungal infection':1,'Allergy':2,'GERD':3,'Chronic cholestasis':4,'Drug Reaction':5,
'Peptic ulcer diseae':6,'AIDS':7,'Diabetes ':8,'Gastroenteritis':9,'Bronchial Asthma':10,'Hypertension ':11,
'Migraine':12,'Cervical spondylosis':13,
'Paralysis (brain hemorrhage)':14,'Jaundice':15,'Malaria':16,'Chicken pox':17,'Dengue':18,'Typhoid':19,'hepatitis A':20,
'Hepatitis B':21,'Hepatitis C':22,'Hepatitis D':23,'Hepatitis E':24,'Alcoholic hepatitis':25,'Tuberculosis':26,
'Common Cold':27,'Pneumonia':28,'Dimorphic hemmorhoids(piles)':29,'Heart attack':30,'Varicose veins':31,'Hypothyroidism':32,
'Hyperthyroidism':33,'Hypoglycemia':34,'Osteoarthristis':35,'Arthritis':36,
'(vertigo) Paroymsal  Positional Vertigo':37,'Acne':38,'Urinary tract infection':39,'Psoriasis':40,
'Impetigo':41}},inplace=True)

X_test= tr[l1]
y_test = tr[["prognosis"]]
np.ravel(y_test)

# TRAINING DATA
df=pd.read_csv("Training.csv")

df.replace({'prognosis':{'Fungal infection T':0,'Fungal infection':1,'Allergy':2,'GERD':3,'Chronic cholestasis':4,'Drug Reaction':5,
'Peptic ulcer diseae':6,'AIDS':7,'Diabetes ':8,'Gastroenteritis':9,'Bronchial Asthma':10,'Hypertension ':11,
'Migraine':12,'Cervical spondylosis':13,
'Paralysis (brain hemorrhage)':14,'Jaundice':15,'Malaria':16,'Chicken pox':17,'Dengue':18,'Typhoid':19,'hepatitis A':20,
'Hepatitis B':21,'Hepatitis C':22,'Hepatitis D':23,'Hepatitis E':24,'Alcoholic hepatitis':25,'Tuberculosis':26,
'Common Cold':27,'Pneumonia':28,'Dimorphic hemmorhoids(piles)':29,'Heart attack':30,'Varicose veins':31,'Hypothyroidism':32,
'Hyperthyroidism':33,'Hypoglycemia':34,'Osteoarthristis':35,'Arthritis':36,
'(vertigo) Paroymsal  Positional Vertigo':37,'Acne':38,'Urinary tract infection':39,'Psoriasis':40,
'Impetigo':41}},inplace=True)

X= df[l1]

y = df[["prognosis"]]
np.ravel(y)

def message():
    if (Symptom1.get() == "None" and  Symptom2.get() == "None" and Symptom3.get() == "None" and Symptom4.get() == "None" and Symptom5.get() == "None"):
        messagebox.showinfo("OPPS!!", "ENTER  SYMPTOMS PLEASE")
    else :
        NaiveBayes()

def NaiveBayes():
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.naive_bayes import MultinomialNB
    gnb = MultinomialNB()
    gnb=gnb.fit(X,np.ravel(y))
    from sklearn.metrics import accuracy_score
    y_pred = gnb.predict(X_test.values)
    print("Accuracy")
    print(accuracy_score(y_test, y_pred))
    # print(accuracy_score(y_test, y_pred, normalize=False))
    print("Confusion matrix")
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)
    print(gnb.score(X_test, y_test))

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]

    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = gnb.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(disease[predicted] == disease[a]):
            h='yes'
            break

    if (h=='yes'):
        t3.delete("1.0", END)
        t3.insert(END, disease[a])
    else:
        t3.delete("1.0", END)
        t3.insert(END, "No Disease")

root = Tk()
root.title(" Treatment Prediction ",)
root.configure()

img= PhotoImage(file='Medical Clinic (Facebook Cover).png', master= root)
img_label= Label(root,image=img)
img_label.place(x=0, y= -400)

Symptom1 = StringVar()
Symptom1.set(None)
Symptom2 = StringVar()
Symptom2.set(None)
Symptom3 = StringVar()
Symptom3.set(None)
Symptom4 = StringVar()
Symptom4.set(None)
Symptom5 = StringVar()
Symptom5.set(None)
Symptom6 = StringVar()
Symptom6.set(None)

w2 = Label(root, justify=CENTER, text=" Treatment Prediction", fg="Black")
w2.config(font=("Elephant", 30))
w2.grid(row=1, column=0, columnspan=2, padx=100)

S1Lb = Label(root,  text="Symptom 1")
S1Lb.config(font=("Elephant", 15))
S1Lb.grid(row=7, column=1, pady=10 , sticky=W)

S2Lb = Label(root,  text="Symptom 2")
S2Lb.config(font=("Elephant", 15))
S2Lb.grid(row=8, column=1, pady=10, sticky=W)

S3Lb = Label(root,  text="Symptom 3")
S3Lb.config(font=("Elephant", 15))
S3Lb.grid(row=9, column=1, pady=10, sticky=W)

S4Lb = Label(root,  text="Symptom 4")
S4Lb.config(font=("Elephant", 15))
S4Lb.grid(row=10, column=1, pady=10, sticky=W)

S5Lb = Label(root,  text="Symptom 5")
S5Lb.config(font=("Elephant", 15))
S5Lb.grid(row=11, column=1, pady=10, sticky=W)

S5Lb = Label(root,  text="Symptom 6")
S5Lb.config(font=("Elephant", 15))
S5Lb.grid(row=12, column=1, pady=10, sticky=W)

lr = Button(root, text="Predict",height=2, width=20, command=message)
lr.config(font=("Elephant", 15))
lr.grid(row=9, column=2,pady=0)

OPTIONS = sorted(l1)

S1En = OptionMenu(root, Symptom1,*OPTIONS)
S1En.grid(row=7, column=1)

S2En = OptionMenu(root, Symptom2,*OPTIONS)
S2En.grid(row=8, column=1)

S3En = OptionMenu(root, Symptom3,*OPTIONS)
S3En.grid(row=9, column=1)

S4En = OptionMenu(root, Symptom4,*OPTIONS)
S4En.grid(row=10, column=1)

S5En = OptionMenu(root, Symptom5,*OPTIONS)
S5En.grid(row=11, column=1)

S5En = OptionMenu(root, Symptom6,*OPTIONS)
S5En.grid(row=12, column=1)

t3 = Text(root, height=2, width=30)
t3.config(font=("Elephant", 20))
t3.grid(row=20, column=1 , padx=10)

root.mainloop()
