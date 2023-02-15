#!/usr/bin/env python
# coding: utf-8

# #      Project: Medical Appointment No Shows Analysis

# # 

# #### Project: Medical Appointment No Shows Analysis
#     Table of Contents
#     Introduction
#     Data Wrangling
#     Exploratory Data Analysis
#     Conclusions

# In[3]:


#pwd


# ##### About Data -
# 
# This is an analysis of the No-show appointments datasets. The data contains 110,527 medical appointments record with its 14 associated variables (characteristics). The most important one is if a patient show-up or does not show up to the appointment. The analysis was made to find if a choosing set of variables might likely be a predictor of whether a patient shows up for their appointment or not.

# ##### Variable Description 
# The dataset contains a total of 14 variables. However as the analysis progressed the number of variables were trimmed down. The variables that remained were those that the analyst thought might be of importance to the analysis. The total variables of the dataset by default are as follows:
# 
# 1.  PatientId
# 2.  AppointmentID
# 3.  Gender
# 4.  ScheduledDay
# 5.  AppointmentDay
# 6.  Age
# 7.  Neighbourhood
# 8.  Scholarship
# 9.  Hipertension
# 10. Diabetes
# 11. Alcoholism
# 12. Handcap
# 13. SMS_received
# 14. No-show
# 
# ###### While the trimmed down dataset includes the following columns
# 
# Introduction
# About
# This is an analysis of the No-show appointments datasets. The data contains 110,527 medical appointments record with its 14 associated variables (characteristics). The most important one is if a patient show-up or does not show up to the appointment. The analysis was made to find if a choosing set of variables might likely be a predictor of wether a patient shows up for their appointment or not.
# 
# Variable Description
# The dataset contains a total of 14 variables. However as the analysis progressed the number of variables were trimmed down. The variables that remained were those that the analyst thought might be of importance to the analysis. The total variables of the dataset by default are as follows:
# 
# PatientId
# AppointmentID
# Gender
# ScheduledDay
# AppointmentDay
# Age
# Neighbourhood
# Scholarship
# Hipertension
# Diabetes
# Alcoholism
# Handcap
# SMS_received
# No-show
# While the trimmed down dataset includes the following columns
# 
# 1. Gender > The gender of the patient
# 2. Scheduled day > The day the appointment was setup
# 3. Appointment day > The main appointment day
# 4. Age > Age of the patient
# 5. Neighbourhood > Location of the hospital
# 6. Scholarship > Whether or not a patient is a welfare        recipient
# 7. Hypertension > Hypertensive status of the patient
# 8. Diabetes > Diabetic status of the patient
# 9. Alcoholism > Whether the patient is an alcoholic
# 10. Handcap > whether the patient is an Handicap
# 10. Sms received > Whether an SMS reminder was sent
# 11. show > Whether the patient showed up for the   appointment

# #### Analysis question
# 
# ######  What factors are important to keep in mind in order        to predict if a patient will show up for their            scheduled appointment ?

# In[4]:


#Library we need to import is - 

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np


# In[5]:


df = pd.read_csv('C:\\Users\\91704\\Downloads\\Data science udemy\\03-Pandas\\KaggleV2-May-2016.csv')


# In[6]:


#df = pd.read_csv('C:\\Users\\91704\\Downloads\\Data science udemy\\03-Pandas\\medical_appoinment_no_show.csv')


# #### Let's understand the data
# 
# This section includes the steps taken to understand the dataset. Also included as part of the session was the data cleaning stage. The cleaning was done to ensure data consistency, correct data formatting, handling duplicates and adjusting columns names for convenience of the analysis

# In[7]:


df.head(3)


# In[8]:


df['No-show'].unique()


# In[9]:


#datatypes of variables
df.info()


# In[10]:


#dimension of data(rows,columns)
df.shape


# In[11]:


# statistical description of dataset
df.describe()


# In[12]:


#variables (column) names
df.columns


# ##### Check for missing values

# In[13]:


df.isna().sum()


# from the above observation it could be observed that
# the orginial dataset has no missing values.

# In[ ]:





# ### Data Cleaning - 

# ##### Adjust column names - convert to lowercase to ensure consistency and typing convenience

# In[14]:


df.columns = pd.Series(df.columns).apply(lambda x:x.lower().replace('-','_').replace('day','_day'))


# ###### Adjusting spellings and confusing column name

# ###### The 'no_show' column was confusing, so it was renamed show.

# In[15]:


#Adjust spelling of 'hypertension'
df.rename(columns={'hipertension':'hypertension','no_show':'show'},inplace=True)


# In[16]:


df.columns


# ##### Remove non useful columns -
# 
# Columns patientid and appointmentid were removed because they were dimmed to be of no relevance to the analysis. This decision was predicated on the fact that the anlaysis does no go into individuals but rather the population as a whole.

# In[17]:


df.drop(['patientid','appointmentid'],axis=1,inplace=True)

df.columns


# ##### Feature engineering
# 
# This section contain adjustment to the dataset variable
# elements for easy manipulation

# In[18]:


df.head(2)


# Below is the adjustments to the show column values made to reflect the change of name of the column.

# In[19]:


# switch 'show' column values to reflect the change of
# name from no_show to show

df.show = df.show.apply(lambda x:'Yes' if x =='No' else 'No')

df.sample(5)


# In[20]:


df.iloc[:,[5,6,7,8,10]].nunique()


# In[21]:


df['show']=df['show'].apply(lambda x:'Yes' if x =='Yes' else 'No')


# In[22]:


df.sample(2)


# In[23]:


df.iloc[:,[5,6,7,8,9,10]].nunique()


# # 

# #####  Value observation

#   The unique values above was observed to be boolean by intention. Therefor the process below changes 1s and 0s to be 'Yes' and 'No

# In[24]:


def change_01_to_str_eq(x):
    return 'Yes' if x==1 else 'No'

target_columns = df.iloc[:,[5,6,7,8,10]].columns
for column in target_columns:
    df[column] = df[column].apply(change_01_to_str_eq)


# In[25]:


df.head()


# In[26]:


df.dtypes


# # 

# ### Exploratory Data Analysis

# The section below contains the exploratory analysis of the dataset. The steps are categorized based on particular question they are answering. Each question contains some data manipulation and plots that is intended to fairly answer the question.

# ###### Can age be an indicator of whether a patient shows up for a medical appointment? 

# ##### 

# In[27]:


df.describe()


# In[28]:


# remove abnormal age
age_df = df.drop(df[df.age<=-1].index,axis=0)
age_df[age_df.age<=-1]


# In[31]:


age_df['age'].hist(figsize=(8,5),bins=10);


# In[31]:


age_df['age'].unique()


#  

# ##### Grouping Age range

# To simplify analysis on age, the age variable is classified in a new column named 'age_label'

# In[32]:


def group_age(x):
    if x<=14:
        return 'child'
    if x>14 and x<=24:
        return 'youth'
    if x > 24 and x <= 64:
        return 'adult'
    if x>64:
        return 'senior'

age_df['age_label'] = age_df['age'].apply(group_age)
age_df.sample(2)


# # 

# ##### Percentage of 'showing up' by the different age groups

# Below is the total observations of each age group

# In[41]:


#totals in age groups

labels = age_df['age_label'].unique()
group_totals = {label: age_df[age_df['age_label']==
label]['age'].size for label in labels}     
group_totals


# In[44]:


## Group dataset of the age groups
age_group_df = age_df.groupby(['show', 'age_label'],as_index=False)['age'].count()

# rename the column
age_group_df.rename(columns = {'age':'ind_count'},inplace=True)

#calcultae proportion in terms of percentage 
def calc_prop(x):
    return round((x['ind_count']/group_totals[x['age_label']])*100,2) #....Doubt

age_group_df['%_of_total'] = age_group_df.apply(calc_prop,axis=1)


#Filter df to patient that 'show'
age_group_show_df = age_group_df[age_group_df['show'] == 'Yes']


# In[45]:


age_group_df


# In[46]:


plt.figure(figsize = (10,6))
plt.ylim(0,100)

ax = sns.barplot(x = 'age_label',y = '%_of_total',hue='show',
        data = age_group_df,palette='hls',
                 order = ['child','youth','adult','senior'],
                 capsize = 0.05, saturation = 8)
plt.xlabel('age group')
plt.ylabel('show up percentage')
plt.title('Percent proportion of appointment show up among age groups')
plt.grid(axis = 'y')

for i in ax.containers:
          ax.bar_label(i)

plt.show()


# # 

# ##### can the gender of a patient be a determinant of the likelihood of a pateint showing up for a medical appointment

# ###### These question explores the likelihood of the impact of gender as a predictor of a patient showing up for appointment.

# In[43]:


#Gender Totals

total_male = df[df['gender']=='M']['gender'].size
total_female = df[df['gender']=='F']['gender'].size

gender_grouped_df = df.groupby(['gender','show'],as_index
    = False).count().iloc[:,:3]

gender_grouped_df.rename(columns={'scheduled_day':'count'},
                        inplace = True)

gender_grouped_df


# In[44]:


gender_grouped_df['%_of_total'] = gender_grouped_df.apply(
lambda x: round((x['count']/total_female)* 100,2)
if x['gender']=='F' else round ((x['count']/total_male)*
                               100,2),axis=1)

#make gender label more intuitive
gender_grouped_df['gender'] = gender_grouped_df['gender'].apply(lambda x: 'Female' if x == 'F' else 'Male')
gender_grouped_df


# In[45]:


#plot

plt.figure(figsize=(10,6))
plt.ylim(0,100)

ax = sns.barplot(x = 'gender',y = '%_of_total',hue = 'show'
,data = gender_grouped_df, palette ='hls',capsize=0.05,
                saturation =8)

plt.xlabel('Gender')
plt.ylabel('Show up percentage')
plt.title('percent proportion of appointment show up by gender')
plt.grid(axis = 'y')

for i in ax.containers:
          ax.bar_label(i,)
plt.show()


# # 

# ##### Does the appointment day of the week determine if a pateint will show up ?

# These question seeks to explore the likely impact of the 
# day of the week of the appointment on the likelihood of the
# pateint showing up. To acheive this, another column containing
# day of the week value was added

#  

# In[47]:


# Add weekday column


# In[48]:


df.head(2)


# In[49]:


def convert_day_name(d):
    '''
    This function converts a date type string to a day of week
       return str --> eg 'Friday'
    '''
    new_date = pd.Timestamp(d)
    return new_date.day_name()

convert_day_name('2016-04-29T18:38:08Z')


# In[50]:


# Create day of the week column
df['ap_day_of_week'] = df.appointment_day.apply(convert_day_name)

df.sample(2)


# In[ ]:





# In[51]:


#Group by appoinment day of the week

day_group_df = df.groupby(['ap_day_of_week','show'],
        as_index = False)['age'].count()

#filter to only appointments that show
# day_group_df = day_group_df[day_group_df.show=='Yes']

#rename 'age'
day_group_df.rename(columns = {'age':'show_count'},
                   inplace = True)

day_group_df


# # 

# Total rows for each of day of the week

# In[52]:


day_totals = {day:df[df['ap_day_of_week']==day]['age'].size
        for day in day_group_df['ap_day_of_week']}

day_totals


# In[57]:


# calculation proportion function and add percentage column

def calc_prop(x):
    return  round((x['show_count']/day_totals[x.ap_day_of_week])*100,2)

day_group_df['%_of_totals'] = day_group_df.apply(calc_prop,
axis = 1)

day_group_df


# In[60]:


#plot for label and proportion
plt.figure(figsize=(10,6))
plt.ylim(0,100)

ax = sns.barplot(x='ap_day_of_week',y='%_of_totals',hue='show',
            data=day_group_df,palette='hls',capsize=0.05,
                order=['Monday','Tuesday','Wednesday','Thursday',
            'Saturday'],saturation=8)

plt.xlabel('Week Days')
plt.ylabel('Show up percentage')
plt.title('Percent proportion of appointment show up by day of th week')
plt.grid(axis='y')

for i in ax.containers:
    ax.bar_label(i,)

plt.show()


# # 

# ##### Does SMS reminders influence showing up ?

# ##### The aim of this question is the exploration of the possible impact of sms reminder on the likelihood of a pateint showing up for appointment.

# In[49]:


## Recieved sms and showed up grouped
sms_group_df = df.groupby(['sms_received','show'],
        as_index=False).age.count()
sms_group_df


# In[63]:


#get totals

received_total = sms_group_df[sms_group_df.sms_received == 'Yes']['age'].sum()
no_received_total = sms_group_df[sms_group_df['sms_received']=='No']['age'].sum()

totals = {'Yes': received_total, 'No': no_received_total}                              

totals


# In[64]:


## add percent proportion column
sms_group_df['%_of_total'] = sms_group_df.apply(lambda x:
    round(x.age/totals[x.sms_received],2)*100,axis=1)

sms_group_df


# In[65]:


#plot

labels = ['Receive', 'Not received']

plt.figure(figsize=(10,8))
plt.ylim(0,100)
plt.grid(axis='y')

#plot
ax = sns.barplot(x = 'sms_received',y='%_of_total',hue='show',
    palette='hls',data= sms_group_df,order=['Yes','No'],
                capsize=0.05,saturation=8)

plt.xlabel('SMS received')
plt.ylabel('Show up percentage')
plt.title('Percent Proportion of appointment show up by received')

for i in ax.containers:
    ax.bar_label(i)
    
plt.show()


# # 

# ##### Does the length of time between booking and actual apoointment time effect showing up?

# This question seek to explore whether the time space 
# between scheduling of appointment and the actual 
# appointment time might likely have any effect on showing.
# 
# This is acheived by the addition of two more columns.
# The contact_duration column was added to contain the 
# integer representation of the duration representation of
# the duration(days).While the duration_label was added to groupa 
# and label the days.
# This is done due to the large distributuion of duration
# time

# In[51]:


df.head(1)


# In[52]:



df['contact_duration'] = pd.to_datetime(df
['appointment_day'])- pd.to_datetime(df['scheduled_day'])
    
df['contact_duration'] = df['contact_duration'].apply(
lambda x:x.days)
df['contact_duration'].describe()
# df['contact_duration']


# In[53]:


df['contact_duration'].hist();


# In[54]:


df.contact_duration[df.contact_duration < 0].min()


# In[55]:


df.head(1)


# In[56]:


# explore negative duration

df.contact_duration[df.contact_duration < 0].count()


# In[57]:


# drop duration with negative values

df_filter_bad_duration = df.drop(df[df.contact_duration 
                                   < 0].index)

df_filter_bad_duration['contact_duration'].describe()


# In[58]:


# group duration values

def group_duration(x):
    '''Labels the interval of int x'''
    if x <=1 :
        return '1 day'
    if x > 1 and x <= 7:
        return '1 week'
    if x > 7 and x <= 14:
        return '2 week'
    if x > 14 and x <= 21:
        return '3 week'
    if x > 21 and x <= 28:
        return '1 month'
    if x > 28 and x <= 56:
        return '2 months'
    if x > 56:
        return '3 months and above'
        


# In[112]:


#Group and label duration

df_filter_bad_duration['duration_label'] =df_filter_bad_duration['contact_duration'].apply(group_duration)

df_filter_bad_duration.sample(5)


# Totals of duration grouping

# In[113]:


# duration_count_totals

duration_labels = df_filter_bad_duration.duration_label.value_counts().index
duration_count_totals = {label: df_filter_bad_duration[df_filter_bad_duration.duration_label == label].shape[0] for label in duration_labels}

duration_count_totals


# In[116]:


# duration and show up grouping

duration_group_df = df_filter_bad_duration.groupby([
    'duration_label', 'show'], 
    as_index = False)['contact_duration'].count()

#rename last column for clarity
duration_group_df.rename(columns = 
    {'contact_duration':'duration_count'},inplace = True)

duration_group_df


# In[117]:


# percentage calc functiom

def calc_prop(x):
    return round((x['duration_count']/duration_count_totals
                 [x.duration_label])*100,2)


# In[118]:


# Add a percentage column

duration_group_df['%_of_total'] = duration_group_df.apply(calc_prop,axis =1)

duration_group_df


# In[120]:


# plot

plt.figure(figsize=(10,6))
plt.ylim(0,100)

#bar_height_text(duration_group_df)
#plt.bar(duration_group_df['duration_label'],
#height = duration_group_df['%_of_total'],width=0.7,align='center')
#plt.show()
    
                           
ax =  sns.barplot(x='duration_label',y='%_of_total',hue='show',palette = 'hls',
                  data=duration_group_df,
        order = ['1 day','1 week','2 week', '3 week',
        '1 month','2 months','3 months and above'],capsize=0.05,
                saturation = 8)

plt.xlabel('Duration')
plt.ylabel('show up percentage')
plt.title('percentage proportion of appointment show by schedule and appointment interval')
plt.grid(axis='y')
                           
for i in ax.containers:
    ax.bar_label(i,)
plt.show()                           
                


# # 

# ##### Does the type of ailment/disease determine showing up ?

# This question explores the impact of variables hypertension,
# diabetes,alcoholism on the likelihood of showing up for an
# appointment.

# In[122]:


df.head(2)


# In[128]:


#group base on ailments

hyp_group_df = df.groupby(['hypertension','show'],
                          as_index=False)['age'].count()
diab_group_df = df.groupby(['diabetes','show'],
            as_index=False)['age'].count()
alc_group_df = df.groupby(['alcoholism','show'],
            as_index=False)['age'].count()

#group individual elemnts based on showing up status
#hypertension pateints

hyp_show = hyp_group_df[(hyp_group_df.hypertension=='Yes')
    & (hyp_group_df.show=='Yes')].iloc[0,2]
hyp_no_show = hyp_group_df[(hyp_group_df.hypertension=='Yes')
    & (hyp_group_df.show=='No')].iloc[0,2]

#diabetic patients
diab_show = diab_group_df[(diab_group_df['diabetes']=='Yes')
    & (diab_group_df['show']=='Yes')].iloc[0,2]
diab_no_show = diab_group_df[(diab_group_df.diabetes=='Yes')
        & (diab_group_df.show=='No')].iloc[0,2]

#alcholic pateints
alc_show = alc_group_df[(alc_group_df['alcoholism']=='Yes')
    & (diab_group_df['show']=='Yes')].iloc[0,2]
alc_no_show = diab_group_df[(alc_group_df.alcoholism=='Yes')
        & (diab_group_df.show=='No')].iloc[0,2]


# In[129]:


#values for plots

hyp_sizes = [hyp_show,hyp_no_show]
hyp_labels = ['show','no show']

diab_sizes = [diab_show,diab_no_show]
diab_labels = ['show','no show']

alc_sizes = [alc_show, alc_no_show]
alc_labels = ['show', 'no show']


# In[130]:


#plots

fig1, (ax1,ax2,ax3) = plt.subplots(1,3,figsize = (10,8))
ax1.pie(hyp_sizes, labels = hyp_labels, autopct ='%1.1f%%',
    shadow = True, radius = 3)
ax1.set_title('Hypertension',pad=65)

ax2.pie(diab_sizes,labels=diab_labels,autopct='%1.1f%%',
       shadow=True,radius=3)
ax2.set_title('Diabetics',pad=65)

ax3.pie(alc_sizes,labels=alc_labels,autopct='%1.1f%%',
       shadow=True,radius=3)
ax3.set_title('Alcoholics',pad=65)

fig1.subplots_adjust(wspace=2)

plt.show()


# ### Conclusions

# The dataset contains variables that belong mostly to the norminal level of measurement hence the heavy reliance on grouping and bar chat plots. In the plots, it was obvious that the differences between variables are mostly small. However, a combination of variables, especially those with a bit higher percentage points might show relationships that might likely be considered, with additional analysis, as predictors of a patient showing up. The dataset is short of quantitative variables, so adequately representing statistical correlation was not possible. The analysis relied heavily on categorization and proportion.

# #### Analysis Questions and answers

# #####  - Can age be an indicator of wether a patient shows up for a medical appointment?

# From the dataset, the show up proportion percentage of individuals with age 64 and above (labeled 'seniors') is higher than the rest (84.5%). The lowest proportion percentage are patients within ages 15 and 24 (labeled 'youths'). In between lies 'adults' (80.26%) and 'children' (78.82%). The middle group is relatively close. However, this is not surprising due to the fact that must children aged 14 and below are dependent on adults aged 25 to 64. Using the knowledege that the human body increasingly becomes week with age, it is not surprising that the seniors have higher percentage of show ups. On the other end, patients described as youth are new adults and tend to be a bit more reckless, and might skip more appointements. Given this evidence from the data, there might likely be a correlation between the patients age and the possibility of showing up for the medical appointments.

# ##### - Can the gender of a patient be a determinant of the likelihood of a patient showing up for a medical appointment?

# Analysis of the dataset reveals that the gender of patients recorded cannot be reliably correlated to tendency to show up fo the appointments. The percent proportion of female patients is 79.6 while that of the male patients is 80.03

# ##### - Does the appointment day of the week determine if a patient will show up?

# Analysis of the dataset for this question posits that there is relatively equal percent ratio of showing up for all the seven days in a week. If this findings where to be correlated with the posibility of showing up, there is might be equal chance of patients showing up or not at any day of the week. Although, it was observed from the dataset that saturdays have the lowest show up rate. Emphasis was not not placed on it due to the fact that on 39 observations were recorded on the day in total, which is small compared with the thousands in others.

# ##### - Does SMS reminders influence showing up?

# By far the most interesting findings of this analysis. The percent proportion of those that recieved SMS reminders and showed up is less than those that did not recieved and showed. Conventional logic might dictate that if a reminder is sent, the individual might more likely show up because abstentations have been most often ascribed to forgetfulness. But reverse is the case in the context. In correlation to showing up, reminder is negatively related. Although, these cannot be considered as a validation that not sending SMS can likely increase show up rate.

# ##### -Does the length of time between booking and actual appointment time affect showing up?

# In the analysis for this question, the dataset set seem to suggest that appointments schedule for a day or less have the greater proportion of been fulfilled. The percent proportion of a day or less is significatly higher and can be correlated with the likelihood of showing up.

# #### Limitations

# The analysis was done based purely on the categorical variables that are dominant in the dataset. Analysis of categorical values have low reliability because of the limited analytical techniques that could be applied to them. Hence, the dominant method adopted for the study is the percent proportion analysis. This technique suffers from a range of issues, one of which is it doesnt fully capture the total number of individual observations that make up a particular group. For example, a group with a fewer set of values can give the illusion of a very high or very low percentage compared to other groups with higher total values. Also, the dataset was missing a number of key variables that could have likely increased the reliability of the analysis result. Variables like distance from the patient to the hospital, educational background of the patient, economic status of the neighborhood, etc. are other key variables that could have shed more light on the factors that could determine keeping up with medical appointments by patients

# #### Conclusion

# To answer the analysis question, a host of variables need to be considered as likely factors to predict a patient showing up for appointments. Most of the variables analysed were found to not be of significant proportion as to warrant a reliable conclusion. These and the absence of quantitative variables for real correlation analysis further affects the dependability of the results. However, from the analysis of the dataset, it is believed that considering multiple variables together could be potential indicators. The stand out variables with noticeable percent proportion are age and duration of time between contact and appointment. Positing on this, and considering potentially more data analysis on causation, an individual with advance age and appointment set less than a day, has a greater probability of showing up. Age and duration can be deemed positively correlated to showing up.
