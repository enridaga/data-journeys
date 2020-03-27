
import numpy as np 
import pandas as pd

# Loading the multiple choices dataset, we will not look to the free form data on this study
mc = pd.read_csv('../input/kaggle-survey-2018/multipleChoiceResponses.csv', low_memory=False)

# Separating questions from answers
# This Series stores all questions
mcQ = mc.iloc[0,:]
# This DataFrame stores all answers
mcA = mc.iloc[1:,:]
# removing everyone that took less than 4 minutes or more than 600 minutes to answer the survey
less3 = mcA[round(mcA.iloc[:,0].astype(int) / 60) <= 4].index
mcA = mcA.drop(less3, axis=0)
more300 = mcA[round(mcA.iloc[:,0].astype(int) / 60) >= 600].index
mcA = mcA.drop(more300, axis=0)

# removing gender trolls, because we noticed from other kernels thata there are some ouliers here
gender_trolls = mcA[(mcA.Q1 == 'Prefer to self-describe') | (mcA.Q1 == 'Prefer not to say')].index
mcA = mcA.drop(list(gender_trolls), axis=0)

# removing student trolls, because a student won't make more than 250k a year.
student_trolls = mcA[((mcA.Q6 == 'Student') & (mcA.Q9 > '500,000+')) | \
                     ((mcA.Q6 == 'Student') & (mcA.Q9 > '400-500,000')) | \
                     ((mcA.Q6 == 'Student') & (mcA.Q9 > '300-400,000')) | \
                     ((mcA.Q6 == 'Student') & (mcA.Q9 > '250-300,000'))].index
mcA = mcA.drop(list(student_trolls), axis=0)

# dropping all NaN and I do not wish to disclose my approximate yearly compensation, because we are only interested in respondents that revealed their earnings
mcA = mcA[~mcA.Q9.isnull()].copy()
not_disclosed = mcA[mcA.Q9 == 'I do not wish to disclose my approximate yearly compensation'].index
mcA = mcA.drop(list(not_disclosed), axis=0)
# Creating a table with personal data
personal_data = mcA.iloc[:,:13].copy()

# renaming columns
cols = ['survey_duration', 'gender', 'gender_text', 'age', 'country', 'education_level', 'undergrad_major', 'role', 'role_text',
        'employer_industry', 'employer_industry_text', 'years_experience', 'yearly_compensation']
personal_data.columns = cols

# Drop text and survey_duration columns 
personal_data.drop(['survey_duration', 'gender_text', 'role_text', 'employer_industry_text'], axis=1, inplace=True)

personal_data.head(3)
from pandas.api.types import CategoricalDtype

# transforming compensation into category type and ordening the values
categ = ['0-10,000', '10-20,000', '20-30,000', '30-40,000', '40-50,000',
         '50-60,000', '60-70,000', '70-80,000', '80-90,000', '90-100,000',
         '100-125,000', '125-150,000', '150-200,000', '200-250,000', '250-300,000',
         '300-400,000', '400-500,000', '500,000+']
cat_type = CategoricalDtype(categories=categ, ordered=True)
personal_data.yearly_compensation = personal_data.yearly_compensation.astype(cat_type)
# Doing this we are transforming the category "I do not wish to disclose my approximate yearly compensation" into NaN

# transforming age into category type and sorting the values
categ = ['18-21', '22-24', '25-29', '30-34', '35-39', '40-44', 
         '45-49', '50-54', '55-59', '60-69', '70-79', '80+']
cat_type = CategoricalDtype(categories=categ, ordered=True)
personal_data.age = personal_data.age.astype(cat_type)

# transforming years of experience into category type and sorting the values
categ = ['0-1', '1-2', '2-3', '3-4', '4-5', '5-10',
         '10-15', '15-20', '20-25', '25-30', '30+']
cat_type = CategoricalDtype(categories=categ, ordered=True)
personal_data.years_experience = personal_data.years_experience.astype(cat_type)

# transforming education level into category type and sorting the values
categ = ['No formal education past high school', 'Some college/university study without earning a bachelor’s degree',
         'Professional degree', 'Bachelor’s degree', 'Master’s degree', 'Doctoral degree', 'I prefer not to answer']
cat_type = CategoricalDtype(categories=categ, ordered=True)
personal_data.education_level = personal_data.education_level.astype(cat_type)
personal_data.yearly_compensation.value_counts(dropna=False, sort=False)
compensation = personal_data.yearly_compensation.str.replace(',', '').str.replace('500000\+', '500-500000').str.split('-')
personal_data['yearly_compensation_numerical'] = compensation.apply(lambda x: (int(x[0]) * 1000 + int(x[1]))/ 2) / 1000 # it is calculated in thousand dollars
print('Dataset Shape: ', personal_data.shape)
personal_data.head(3)
# Finding the compensation that separates the Top 20% most welll paid from the Bottom 80%
top20flag = personal_data.yearly_compensation_numerical.quantile(0.8)
top20flag
# Creating a flag to identify who belongs to the Top 20%
personal_data['top20'] = personal_data.yearly_compensation_numerical > top20flag

# creating data for future mapping of values
top20 = personal_data.groupby('yearly_compensation', as_index=False)['top20'].min()
# Some helper functions to make our plots cleaner with Plotly
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
init_notebook_mode(connected=True)


def gen_xaxis(title):
    """
    Creates the X Axis layout and title
    """
    xaxis = dict(
            title=title,
            titlefont=dict(
                color='#AAAAAA'
            ),
            showgrid=False,
            color='#AAAAAA',
            )
    return xaxis


def gen_yaxis(title):
    """
    Creates the Y Axis layout and title
    """
    yaxis=dict(
            title=title,
            titlefont=dict(
                color='#AAAAAA'
            ),
            showgrid=False,
            color='#AAAAAA',
            )
    return yaxis


def gen_layout(charttitle, xtitle, ytitle, lmarg, h, annotations=None):  
    """
    Creates whole layout, with both axis, annotations, size and margin
    """
    return go.Layout(title=charttitle, 
                     height=h, 
                     width=800,
                     showlegend=False,
                     xaxis=gen_xaxis(xtitle), 
                     yaxis=gen_yaxis(ytitle),
                     annotations = annotations,
                     margin=dict(l=lmarg),
                    )


def gen_bars(data, color, orient):
    """
    Generates the bars for plotting, with their color and orient
    """
    bars = []
    for label, label_df in data.groupby(color):
        if orient == 'h':
            label_df = label_df.sort_values(by='x', ascending=True)
        if label == 'a':
            label = 'lightgray'
        bars.append(go.Bar(x=label_df.x,
                           y=label_df.y,
                           name=label,
                           marker={'color': label},
                           orientation = orient
                          )
                   )
    return bars


def gen_annotations(annot):
    """
    Generates annotations to insert in the chart
    """
    if annot is None:
        return []
    
    annotations = []
    # Adding labels
    for d in annot:
        annotations.append(dict(xref='paper', x=d['x'], y=d['y'],
                           xanchor='left', yanchor='bottom',
                           text= d['text'],
                           font=dict(size=13,
                           color=d['color']),
                           showarrow=False))
    return annotations


def generate_barplot(text, annot_dict, orient='v', lmarg=120, h=400):
    """
    Generate the barplot with all data, using previous helper functions
    """
    layout = gen_layout(text[0], text[1], text[2], lmarg, h, gen_annotations(annot_dict))
    fig = go.Figure(data=gen_bars(barplot, 'color', orient=orient), layout=layout)
    return iplot(fig)
# Counting the quantity of respondents per compensation
barplot = personal_data.yearly_compensation.value_counts(sort=False).to_frame().reset_index()
barplot.columns = ['yearly_compensation', 'qty']

# mapping back to get top 20% label
barplot = barplot.merge(top20, on='yearly_compensation')
barplot.columns = ['x', 'y', 'top20']

# apply color for top 20% and bottom 80%
barplot['color'] = barplot.top20.apply(lambda x: 'mediumaquamarine' if x else 'lightgray') 

# Create title and annotations
title_text = ['<b>How Much Does Kagglers Get Paid?</b>', 'Yearly Compensation (USD)', 'Quantity of Respondents']
annotations = [{'x': 0.06, 'y': 2200, 'text': '80% of respondents earn up to USD 90k','color': 'gray'},
              {'x': 0.51, 'y': 1100, 'text': '20% of respondents earn more than USD 90k','color': 'mediumaquamarine'}]

# call function for plotting
generate_barplot(title_text, annotations)
# creating masks to identify students and not students
is_student_mask = (personal_data['role'] == 'Student') | (personal_data['employer_industry'] == 'I am a student')
not_student_mask = (personal_data['role'] != 'Student') & (personal_data['employer_industry'] != 'I am a student')

# Counting the quantity of respondents per compensation (where is student)
barplot = personal_data[is_student_mask].yearly_compensation.value_counts(sort=False).to_frame().reset_index()
barplot.columns = ['yearly_compensation', 'qty']

# mapping back to get top 20%
barplot.columns = ['x', 'y',]
barplot['highlight'] = barplot.x != '0-10,000'

# applying color
barplot['color'] = barplot.highlight.apply(lambda x: 'lightgray' if x else 'crimson')

# title and annotations
title_text = ['<b>Do Students Get Paid at All?</b><br><i>only students</i>', 'Yearly Compensation (USD)', 'Quantity of Respondents']
annotations = [{'x': 0.06, 'y': 1650, 'text': '75% of students earn up to USD 10k','color': 'crimson'}]

# ploting
generate_barplot(title_text, annotations)
# Finding the compensation that separates the Top 20% most welll paid from the Bottom 80% (without students)
top20flag_no_students = personal_data[not_student_mask].yearly_compensation_numerical.quantile(0.8)
top20flag_no_students
# Creating a flag for Top 20% when there are no students in the dataset
personal_data['top20_no_students'] = personal_data.yearly_compensation_numerical > top20flag_no_students

# creating data for future mapping of values
top20 = personal_data[not_student_mask].groupby('yearly_compensation', as_index=False)['top20_no_students'].min()

# Counting the quantity of respondents per compensation (where is not student)
barplot = personal_data[not_student_mask].yearly_compensation.value_counts(sort=False).to_frame().reset_index()
barplot.columns = ['yearly_compensation', 'qty']

# mapping back to get top 20%
barplot = barplot.merge(top20, on='yearly_compensation')
barplot.columns = ['x', 'y', 'top20']
barplot['color'] = barplot.top20.apply(lambda x: 'mediumaquamarine' if x else 'lightgray')

title_text = ['<b>How Much Does Kagglers Get Paid?</b><br><i>without students</i>', 'Yearly Compensation (USD)', 'Quantity of Respondents']
annotations = [{'x': 0.06, 'y': 1600, 'text': '80% of earn up to USD 100k','color': 'gray'},
              {'x': 0.56, 'y': 800, 'text': '20% of earn more than USD 100k','color': 'mediumaquamarine'}]

generate_barplot(title_text, annotations)
# Creating a helper function to generate lineplot
def gen_lines(data, colorby):
    """
    Generate the lineplot with data
    """
    if colorby == 'top20': 
        colors = {False: 'lightgray',
                  True: 'mediumaquamarine'}
    else:
        colors = {False: 'lightgray',
                  True: 'deepskyblue'}

    traces = []
    for label, label_df in data.groupby(colorby):
        traces.append(go.Scatter(
                    x=label_df.x,
                    y=label_df.y,
                    mode='lines+markers+text',
                    line={'color': colors[label], 'width':2},
                    connectgaps=True,
                    text=label_df.y.round(),
                    hoverinfo='none',
                    textposition='top center',
                    textfont=dict(size=12, color=colors[label]),
                    marker={'color': colors[label], 'size':8},
                   )
                   )
    return traces
# Grouping data to get compensation per gender of Top20% and Bottom 80%
barplot = personal_data[not_student_mask].groupby(['gender', 'top20_no_students'], as_index=False)['yearly_compensation_numerical'].mean()
barplot = barplot[(barplot['gender'] == 'Female') | (barplot['gender'] == 'Male')]
barplot.columns = ['x', 'gender', 'y']

# Creates annotations
annot_dict = [{'x': 0.05, 'y': 180, 'text': 'The top 20% men are almost 12% better paid than the top 20% woman','color': 'deepskyblue'},
              {'x': 0.05, 'y': 60, 'text': 'At the bottom 80% there is almost no difference in payment','color': 'gray'}]

# Creates layout
layout = gen_layout('<b>What is the gender difference in compensation at the top 20%?</b><br><i>without students</i>', 
                    'Gender', 
                    'Average Yearly Compensation (USD)',
                    120, 
                    400,
                    gen_annotations(annot_dict)
                    )
# Make plot
fig = go.Figure(data=gen_lines(barplot, 'gender'), 
                layout=layout)
iplot(fig, filename='color-bar')
# Calculates compensation per education level
barplot = personal_data[not_student_mask].groupby(['education_level'], as_index=False)['yearly_compensation_numerical'].mean()
barplot['no_college'] = (barplot.education_level == 'No formal education past high school') | \
                        (barplot.education_level == 'Doctoral degree')

# creates a line break for better visualisation
barplot.education_level = barplot.education_level.str.replace('study without', 'study <br> without')

barplot.columns = ['y', 'x', 'no_college']
barplot = barplot.sort_values(by='x', ascending=True)
barplot['color'] = barplot.no_college.apply(lambda x: 'coral' if x else 'a')

# Add title and annotations
title_text = ['<b>Impact of Formal Education on Compenstaion</b><br><i>without students</i>', 'Average Yearly Compensation (USD)', 'Level of Education']
annotations = []

generate_barplot(title_text, annotations, orient='h', lmarg=300)
# Calculates compensation per industry
barplot = personal_data[not_student_mask].groupby(['employer_industry'], as_index=False)['yearly_compensation_numerical'].mean()

# Flags the top 5 industries to add color
barplot['best_industries'] = (barplot.employer_industry == 'Medical/Pharmaceutical') | \
                             (barplot.employer_industry == 'Insurance/Risk Assessment') | \
                             (barplot.employer_industry == 'Military/Security/Defense') | \
                             (barplot.employer_industry == 'Hospitality/Entertainment/Sports') | \
                             (barplot.employer_industry == 'Accounting/Finance')

barplot.columns = ['y', 'x', 'best_industries']
barplot = barplot.sort_values(by='x', ascending=True)
barplot['color'] = barplot.best_industries.apply(lambda x: 'darkgoldenrod' if x else 'a')

title_text = ['<b>Average Compensation per Industry | Top 5 in Color</b><br><i>without students</i>', 'Average Yearly Compensation (USD)', 'Industry']
annotations = []

generate_barplot(title_text, annotations, orient='h', lmarg=300, h=600)
# Calculates compensation per role
barplot = personal_data[not_student_mask].groupby(['role'], as_index=False)['yearly_compensation_numerical'].mean()

# Flags the top 5 roles to add color
barplot['role_highlight'] = (barplot.role == 'Data Scientist') | \
                        (barplot.role == 'Product/Project Manager') | \
                        (barplot.role == 'Consultant') | \
                        (barplot.role == 'Data Journalist') | \
                        (barplot.role == 'Manager') | \
                        (barplot.role == 'Principal Investigator') | \
                        (barplot.role == 'Chief Officer')

barplot.columns = ['y', 'x', 'role_highlight']
barplot = barplot.sort_values(by='x', ascending=True)
barplot['color'] = barplot.role_highlight.apply(lambda x: 'mediumvioletred' if x else 'lightgray')

title_text = ['<b>Average Compensation per Role | Top 7 in Color</b><br><i>without students</i>', 'Average Yearly Compensation (USD)', 'Job Title']
annotations = [{'x': 0.6, 'y': 11.5, 'text': 'The first step into the ladder<br>of better compensation is<br>becoming a Data Scientist','color': 'mediumvioletred'}]

generate_barplot(title_text, annotations, orient='h', lmarg=300, h=600)
# Replacing long country names
personal_data.country = personal_data.country.str.replace('United Kingdom of Great Britain and Northern Ireland', 'United Kingdom')
personal_data.country = personal_data.country.str.replace('United States of America', 'United States')
personal_data.country = personal_data.country.str.replace('I do not wish to disclose my location', 'Not Disclosed')
personal_data.country = personal_data.country.str.replace('Iran, Islamic Republic of...', 'Iran')
personal_data.country = personal_data.country.str.replace('Hong Kong \(S.A.R.\)', 'Hong Kong')
personal_data.country = personal_data.country.str.replace('Viet Nam', 'Vietnam')
personal_data.country = personal_data.country.str.replace('Republic of Korea', 'South Korea')

# Calculates compensation per country
barplot = personal_data[not_student_mask].groupby(['country'], as_index=False)['yearly_compensation_numerical'].mean()

# Flags the top 10 countries to add color
barplot['country_highlight'] = (barplot.country == 'United States') | \
                               (barplot.country == 'Switzerland') | \
                               (barplot.country == 'Australia') | \
                               (barplot.country == 'Israel') | \
                               (barplot.country == 'Denmark') | \
                               (barplot.country == 'Canada') | \
                               (barplot.country == 'Hong Kong') | \
                               (barplot.country == 'Norway') | \
                               (barplot.country == 'Ireland') | \
                               (barplot.country == 'United Kingdom')

barplot.columns = ['y', 'x', 'country_highlight']
barplot = barplot.sort_values(by='x', ascending=True)
barplot['color'] = barplot.country_highlight.apply(lambda x: 'mediumseagreen' if x else 'lightgray')

title_text = ['<b>Average Compensation per Country - Top 10 in Color</b><br><i>without students</i>', 'Average Yearly Compensation (USD)', 'Country']
annotations = []

generate_barplot(title_text, annotations, orient='h', lmarg=300, h=1200)
# Loading the cost of living
cost_living = pd.read_csv('../input/cost-of-living-per-country/cost_of_living.csv')
cost_living.columns = ['ranking', 'country', 'price_index']
cost_living.head()
# joining both tables
personal_data = personal_data.merge(cost_living, on='country') # doing an inner join to avoid nans on normalized compensation

# calculating the normalized compensation
personal_data['normalized_compensation'] = personal_data.yearly_compensation_numerical / personal_data.price_index * 10
personal_data['normalized_compensation'] = personal_data['normalized_compensation'].round() * 10
# recreating masks
is_student_mask = (personal_data['role'] == 'Student') | (personal_data['employer_industry'] == 'I am a student')
not_student_mask = (personal_data['role'] != 'Student') & (personal_data['employer_industry'] != 'I am a student')
# Calculates compensation per country
barplot = personal_data[not_student_mask].groupby(['country'], as_index=False)['normalized_compensation'].mean()

# Flags the top 10 countries to add color
barplot['country_highlight'] = (barplot.country == 'United States') | \
                               (barplot.country == 'Australia') | \
                               (barplot.country == 'Israel') | \
                               (barplot.country == 'Switzerland') | \
                               (barplot.country == 'Canada') | \
                               (barplot.country == 'Tunisia') | \
                               (barplot.country == 'Germany') | \
                               (barplot.country == 'Denmark') | \
                               (barplot.country == 'Colombia') | \
                               (barplot.country == 'South Korea')

barplot.columns = ['y', 'x', 'country_highlight']
barplot = barplot.sort_values(by='x', ascending=True)
barplot['color'] = barplot.country_highlight.apply(lambda x: 'mediumseagreen' if x else 'lightgray')

title_text = ['<b>Normalized Average Compensation per Country - Top 10 in Color</b><br><i>without students</i>', 
              'Normalized Average Yearly Compensation (USD)', 'Country']
annotations = []

generate_barplot(title_text, annotations, orient='h', lmarg=300, h=1200)
# Defining the threshold for top 20% most paid
top20_tresh = personal_data.normalized_compensation.quantile(0.8)
personal_data['top20'] = personal_data.normalized_compensation > top20_tresh

# creating data for future mapping of values
top20 = personal_data.groupby('normalized_compensation', as_index=False)['top20'].min()

# Calculates respondents per compensation
barplot = personal_data.normalized_compensation.value_counts(sort=False).to_frame().reset_index()
barplot.columns = ['normalized_compensation', 'qty']

# mapping back to get top 20% and 50%
barplot = barplot.merge(top20, on='normalized_compensation')
barplot.columns = ['x', 'y', 'top20']
barplot['color'] = barplot.top20.apply(lambda x: 'mediumaquamarine' if x else 'lightgray')

title_text = ['<b>How Much Does Kagglers Get Paid?<br></b><i>normalized by cost of living</i>', 'Normalized Yearly Compensation', 'Quantity of Respondents']
annotations = [{'x': 0.1, 'y': 1000, 'text': '20% Most well paid','color': 'mediumaquamarine'}]

generate_barplot(title_text, annotations)
# First we store all answers in a dict
answers = {'Q1': mcA.iloc[:,1],
           'Q2': mcA.iloc[:,3],
           'Q3': mcA.iloc[:,4],
           'Q4': mcA.iloc[:,5],
           'Q5': mcA.iloc[:,6],
           'Q6': mcA.iloc[:,7],
           'Q7': mcA.iloc[:,9],
           'Q8': mcA.iloc[:,11],
           'Q9': mcA.iloc[:,12],
           'Q10': mcA.iloc[:,13],
           'Q11': mcA.iloc[:,14:21],
           'Q12': mcA.iloc[:,22],
           'Q13': mcA.iloc[:,29:44],
           'Q14': mcA.iloc[:,45:56],
           'Q15': mcA.iloc[:,57:64],
           'Q16': mcA.iloc[:,65:83],
           'Q17': mcA.iloc[:,84],
           'Q18': mcA.iloc[:,86],
           'Q19': mcA.iloc[:,88:107],
           'Q20': mcA.iloc[:,108],
           'Q21': mcA.iloc[:,110:123],
           'Q22': mcA.iloc[:,124],
           'Q23': mcA.iloc[:,126],
           'Q24': mcA.iloc[:,127],
           'Q25': mcA.iloc[:,128],
           'Q26': mcA.iloc[:,129],
           'Q27': mcA.iloc[:,130:150],
           'Q28': mcA.iloc[:,151:194],
           'Q29': mcA.iloc[:,195:223],
           'Q30': mcA.iloc[:,224:249],
           'Q31': mcA.iloc[:,250:262],
           'Q32': mcA.iloc[:,263],
           'Q33': mcA.iloc[:,265:276],
           'Q34': mcA.iloc[:, 277:283],
           'Q35': mcA.iloc[:, 284:290],
           'Q36': mcA.iloc[:,291:304],
           'Q37': mcA.iloc[:,305],
           'Q38': mcA.iloc[:,307:329],
           'Q39': mcA.iloc[:,330:332],
           'Q40': mcA.iloc[:,332],
           'Q41': mcA.iloc[:,333:336],
           'Q42': mcA.iloc[:,336:341],
           'Q43': mcA.iloc[:,342],
           'Q44': mcA.iloc[:,343:348],
           'Q45': mcA.iloc[:,349:355],
           'Q46': mcA.iloc[:,355],
           'Q47': mcA.iloc[:,356:371],
           'Q48': mcA.iloc[:,372],
           'Q49': mcA.iloc[:,373:385],
           'Q50': mcA.iloc[:,386:394]}
# Then we store all questions in another dict
questions = {
'Q1': 'What is your gender?',
'Q2': 'What is your age (# years)?',
'Q3': 'In which country do you currently reside?',
'Q4': 'What is the highest level of formal education that you have attained or plan to attain within the next 2 years?',
'Q5': 'Which best describes your undergraduate major?',
'Q6': 'Select the title most similar to your current role (or most recent title if retired)',
'Q7': 'In what industry is your current employer/contract (or your most recent employer if retired)?',
'Q8': 'How many years of experience do you have in your current role?',
'Q9': 'What is your current yearly compensation (approximate $USD)?',
'Q10': 'Does your current employer incorporate machine learning methods into their business?',
'Q11': 'Select any activities that make up an important part of your role at work',
'Q12': 'What is the primary tool that you use at work or school to analyze data?',
'Q13': 'Which of the following integrated development environments (IDEs) have you used at work or school in the last 5 years?',
'Q14': 'Which of the following hosted notebooks have you used at work or school in the last 5 years?',
'Q15': 'Which of the following cloud computing services have you used at work or school in the last 5 years?',
'Q16': 'What programming languages do you use on a regular basis?',
'Q17': 'What specific programming language do you use most often?',
'Q18': 'What programming language would you recommend an aspiring data scientist to learn first?',
'Q19': 'What machine learning frameworks have you used in the past 5 years?',
'Q20': 'Of the choices that you selected in the previous question, which ML library have you used the most?',
'Q21': 'What data visualization libraries or tools have you used in the past 5 years?',
'Q22': 'Of the choices that you selected in the previous question, which specific data visualization library or tool have you used the most?',
'Q23': 'Approximately what percent of your time at work or school is spent actively coding?',
'Q24': 'How long have you been writing code to analyze data?',
'Q25': 'For how many years have you used machine learning methods (at work or in school)?',
'Q26': 'Do you consider yourself to be a data scientist?',
'Q27': 'Which of the following cloud computing products have you used at work or school in the last 5 years?',
'Q28': 'Which of the following machine learning products have you used at work or school in the last 5 years?',
'Q29': 'Which of the following relational database products have you used at work or school in the last 5 years?',
'Q30': 'Which of the following big data and analytics products have you used at work or school in the last 5 years?',
'Q31': 'Which types of data do you currently interact with most often at work or school?',
'Q32': 'What is the type of data that you currently interact with most often at work or school? ',
'Q33': 'Where do you find public datasets?',
'Q34': 'During a typical data science project at work or school, approximately what proportion of your time is devoted to the following?',
'Q35': 'What percentage of your current machine learning/data science training falls under each category?',
'Q36': 'On which online platforms have you begun or completed data science courses?',
'Q37': 'On which online platform have you spent the most amount of time?',
'Q38': 'Who/what are your favorite media sources that report on data science topics?',
'Q39': 'How do you perceive the quality of online learning platforms and in-person bootcamps as compared to the quality of the education provided by traditional brick and mortar institutions?',
'Q40': 'Which better demonstrates expertise in data science: academic achievements or independent projects? ',
'Q41': 'How do you perceive the importance of the following topics?',
'Q42': 'What metrics do you or your organization use to determine whether or not your models were successful?',
'Q43': 'Approximately what percent of your data projects involved exploring unfair bias in the dataset and/or algorithm?',
'Q44': 'What do you find most difficult about ensuring that your algorithms are fair and unbiased? ',
'Q45': 'In what circumstances would you explore model insights and interpret your models predictions?',
'Q46': 'Approximately what percent of your data projects involve exploring model insights?',
'Q47': 'What methods do you prefer for explaining and/or interpreting decisions that are made by ML models?',
'Q48': 'Do you consider ML models to be "black boxes" with outputs that are difficult or impossible to explain?',
'Q49': 'What tools and methods do you use to make your work easy to reproduce?',
'Q50': 'What barriers prevent you from making your work even easier to reuse and reproduce?',
'top7_job_title': 'Select the title most similar to your current role (or most recent title if retired)',
'job_title_student': 'Select the title most similar to your current role (or most recent title if retired)',
'top10_country': 'In which country do you currently reside?',
'age': 'What is your age (# years)?',
'gender-Male': 'What is your gender?',
'top2_education_level': 'What is the highest level of formal education that you have attained or plan to attain within the next 2 years?',
'top5_industries': 'In what industry is your current employer/contract (or your most recent employer if retired)?',
'industry_student': 'In what industry is your current employer/contract (or your most recent employer if retired)?',
'years_experience': 'How many years of experience do you have in your current role?'}
def normalize_labels(full_label):
    """
    treat labels for new column names
    """
    try:
        label = full_label.split('<>')[1] # split and get second item
    except IndexError:
        label = full_label.split('<>')[0] # split and get first item

    return label

def treat_data(data, idx, tresh):
    """
    Clean and get dumies for columns
    """ 
    # get dummies with a distinct separator
    result = pd.get_dummies(data, prefix_sep='<>', drop_first=False)
    # gets and normalize dummies names
    cols = [normalize_labels(str(x)) for x in result.columns]
    
    # build columns labels with questions
    try:
        Qtext = mcQ['Q{}'.format(idx)]
    except KeyError:
        try:
            Qtext = mcQ['Q{}_Part_1'.format(idx)]
        except KeyError:
            Qtext = mcQ['Q{}_MULTIPLE_CHOICE'.format(idx)]
            
    # Build new columns names
    prefix = 'Q{}-'.format(idx)
    result.columns = [prefix + x for x in cols]
    
    # dropping columns that had less than 10% of answers to avoid overfitting
    percent_answer = result.sum() / result.shape[0]
    for row in percent_answer.iteritems():
        if row[1] < tresh:
            result = result.drop(row[0], axis=1)
        
    return result
# selecting the questions
selected_questions = [1, 2, 3, 4, 6, 7, 8, 10, 11, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26, 29, 31, 36, 38, 40, 42, 47, 48, 49]
treated_data = {}

# Formatting all answers from the selected questions, dropping answers with less than 5%
for sq in selected_questions:
    treated_data['Q{}'.format(sq)] = treat_data(answers['Q{}'.format(sq)], sq, 0.05)   
# Done! Now we are able to rebuild a much cleaner dataset!

# Define target variable
compensation = mcA.Q9.str.replace(',', '').str.replace('500000\+', '500-500000').str.split('-')
mcA['yearly_compensation_numerical'] = compensation.apply(lambda x: (int(x[0]) * 1000 + int(x[1]))/ 2) / 1000 # it is calculated in thousand dollars
clean_dataset = (mcA.yearly_compensation_numerical > 100).reset_index().astype(int)
clean_dataset.columns = ['index', 'top20']

# Join with treated questions
for key, value in treated_data.items():
    value = value.reset_index(drop=True)
    clean_dataset = clean_dataset.join(value, how='left')

clean_dataset = clean_dataset.drop('index', axis=1)

# saving back to csv so others may use it
clean_dataset.to_csv('clean_dataset.csv')

clean_dataset.head()
shape = clean_dataset.shape
print('Our cleaned dataset has {} records and {} features'.format(shape[0], shape[1]))
# Create correlation matrix
correl = clean_dataset.corr().abs()

# Select upper triangle of correlation matrix
upper = correl.where(np.triu(np.ones(correl.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.5
to_drop = [column for column in upper.columns if any(upper[column] > 0.5)]

# Drop features 
clean_dataset_dropped = clean_dataset.drop(to_drop, axis=1)

shape = clean_dataset_dropped.shape
print('After dropping highly correlated features, our has {} records and {} features'.format(shape[0], shape[1]))
print('Dropped features: ', to_drop)
# Finding NANs
df = clean_dataset_dropped.isnull().sum().to_frame()
print('We found {} NaNs on the dataset after treatment'.format(df[df[0] > 0].shape[0]))
from sklearn.model_selection import train_test_split

train, test = train_test_split(clean_dataset_dropped, test_size=0.2, random_state=42)
print('Train Shape:', train.shape)
print('Test Shape:', test.shape)
# Separating X,y train and test sets
ytrain = train['top20'].copy()
Xtrain = train.drop(['top20'], axis=1).copy() # removing both target variables from features

ytest = test['top20'].copy()
Xtest = test.drop(['top20'], axis=1).copy() # removing both target variables from features
# Helper function to help evaluating the model
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def display_scores(predictor, X, y):
    """
    Calculates metrics and display it
    """
    print('\n### -- ### -- ' + str(type(predictor)).split('.')[-1][:-2] + ' -- ### -- ###')
    # Getting the predicted values
    ypred = predictor.predict(X)
    ypred_score = predictor.predict_proba(X)
    
    # calculating metrics
    accuracy = accuracy_score(y, ypred)
    roc = roc_auc_score(y, pd.DataFrame(ypred_score)[1])
    confusion = confusion_matrix(y, ypred)
    
    print('Confusion Matrix: ', confusion)
    print('Accuracy: ', accuracy)
    print('AUC: ', roc)
    
    type1_error = confusion[0][1] / confusion[0].sum() # False Positive - model predicted in top 20%, while it wasn't
    type2_error = confusion[1][0] / confusion[1].sum() # False Negative - model predicted out of top 20%, while it was
    
    print('Type 1 error: ', type1_error)
    print('Type 2 error: ', type2_error)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

rforest = RandomForestClassifier(n_estimators=100, random_state=42)
lreg = LogisticRegression(solver='liblinear', random_state=42)

# Fit the models
rforest.fit(Xtrain, ytrain)
lreg.fit(Xtrain, ytrain)

# Check some metrics
display_scores(rforest, Xtrain, ytrain)
display_scores(lreg, Xtrain, ytrain)
from sklearn.model_selection import cross_val_score

def do_cv(predictor, X, y, cv):
    """
    Executes cross validation and display scores
    """
    print('### -- ### -- ' + str(type(predictor)).split('.')[-1][:-2] + ' -- ### -- ###')
    cv_score = cross_val_score(predictor, X, y, scoring='roc_auc', cv=5)
    print ('Mean AUC score after a 5-fold cross validation: ', cv_score.mean())
    print ('AUC score of each fold: ', cv_score)
    
do_cv(rforest, Xtrain, ytrain, 5)
print('\n ----------------------------- \n')
do_cv(lreg, Xtrain, ytrain, 5)
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler

print('Quantity of samples on each class BEFORE undersampling: ', sorted(Counter(ytrain).items()))
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(Xtrain, ytrain)
print('Quantity of samples on each class AFTER undersampling: ', sorted(Counter(y_resampled).items()))
# refit the model
rforest.fit(X_resampled, y_resampled)
lreg.fit(X_resampled, y_resampled)

# do Cross Validation
do_cv(rforest, Xtrain, ytrain, 5)
display_scores(rforest, Xtrain, ytrain)
print('\n ----------------------------- \n')
do_cv(lreg, Xtrain, ytrain, 5)
display_scores(lreg, Xtrain, ytrain)
display_scores(lreg, Xtest, ytest)
# calculating scores
scores = pd.DataFrame(lreg.predict_proba(Xtest)).iloc[:,1]
scores = pd.DataFrame([scores.values, ytest.values]).transpose()
scores.columns = ['score', 'top20']

# Add histogram data
x0 = scores[scores['top20'] == 0]['score']
x1 = scores[scores['top20'] == 1]['score']

bottom80 = go.Histogram(
    x=x0,
    opacity=0.5,
    marker={'color': 'lightgray'},
    name='Bottom 80%'

)
top20 = go.Histogram(
    x=x1,
    opacity=0.5,
    marker={'color': 'mediumaquamarine'},
    name='Top 20%'   
)

annot_dict = [{'x': 0.2, 'y': 180, 'text': 'The 80% less paid tend<br>to have lower scores','color': 'gray'},
              {'x': 0.75, 'y': 95, 'text': 'Top 20% tend to have<br>higher scores','color': 'mediumaquamarine'}]

layout = gen_layout('<b>Distribution of Scores From the Top 20% and Bottom 80%</b><br><i>test data</i>', 
                    'Probability Score',
                    'Quantity of Respondents',
                    annotations=gen_annotations(annot_dict),
                    lmarg=150, h=400
                    )
layout['barmode'] = 'overlay'

data = [bottom80, top20]
layout = go.Layout(layout)
fig = go.Figure(data=data, layout=layout)

iplot(fig)
from sklearn.metrics import roc_curve

yscore = pd.DataFrame(lreg.predict_proba(Xtest)).iloc[:,1]
fpr, tpr, _ = roc_curve(ytest, yscore)

trace1 = go.Scatter(x=fpr, y=tpr, 
                    mode='lines', 
                    line=dict(color='mediumaquamarine', width=3),
                    name='ROC curve'
                   )

trace2 = go.Scatter(x=[0, 1], y=[0, 1], 
                    mode='lines', 
                    line=dict(color='lightgray', width=1, dash='dash'),
                    showlegend=False)

layout = gen_layout('<b>Receiver Operating Characteristic Curve</b><br><i>test data</i>', 
                    'False Positive Rate',
                    'True Positive Rate',
                    lmarg=50, h=600
                    )


fig = go.Figure(data=[trace1, trace2], layout=layout)

iplot(fig)
def calc_proba(model):
    # calculating scores for the test data
    scores = pd.DataFrame(model.predict_proba(Xtest)).iloc[:,1]
    scores = pd.DataFrame([scores.values, ytest.values]).transpose()
    scores.columns = ['score', 'top20']

    # create 10 evenly spaced bins
    scores['bin'] = pd.cut(scores.score, [-0.01, 0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 1])

    # count number of individuals in Top20% and Bottom80% per bin
    prob = scores.groupby(['bin', 'top20'], as_index=False)['score'].count()
    prob = pd.pivot_table(prob, values='score', index=['bin'], columns=['top20'])

    # calculates the probability
    prob['probability'] = prob[1.0] / (prob[0.0] + prob[1.0])
    return prob['probability']

# Calculates the probabilities of belonging to Top20% per range of score based on test data
calc_proba(lreg).to_frame()
print('Our model\'s intercept is:', lreg.intercept_[0])
# treating the questions just to display better names
features = pd.DataFrame([Xtrain.columns, lreg.coef_[0]]).transpose()
features.columns = ['feature', 'coefficient']
features['abs_coefficient'] = features['coefficient'].abs()
features['question_number'] = features.feature.str.split('-').str[0]
features['answer'] = features.feature.str[3:]
features['answer'] = features.answer.apply(lambda x: x[1:] if x[0] == '-' else x)

features['question'] = features['question_number'].map(questions)


answers_dict = {'age': 'continuous feature',
                'top10_country': 'live at one of the top 10 countries',
                'top7_job_title': 'has one of the top 7 job titles',
               }

features['question'] = features['question_number'].map(questions)
features = features[['question_number', 'question', 'answer', 'coefficient', 'abs_coefficient']]
# Helper functions for building clean plots
def gen_yaxis(title):
    """
    Create y axis
    """
    yaxis=dict(
            title=title,
            titlefont=dict(
                color='#AAAAAA'
            ),
            showgrid=False,
            color='#AAAAAA',
            tickfont=dict(
            size=12,
            color='#444444'
        ),
            )
    return yaxis


def gen_layout(charttitle, xtitle, ytitle, annotations=None, lmarg=120, h=400):  
    """
    Create layout
    """
    return go.Layout(title=charttitle, 
                     height=h, 
                     width=800,
                     showlegend=False,
                     xaxis=gen_xaxis(xtitle), 
                     yaxis=gen_yaxis(ytitle),
                     annotations = annotations,
                     margin=dict(l=lmarg),
                    )

def split_string(string, lenght):
    """
    Split a string adding a line break at each "lenght" words
    """
    result = ''
    idx = 1
    for word in string.split(' '):
        if idx % lenght == 0:
            result = result + '<br>' + ''.join(word)
        else:    
            result = result + ' ' + ''.join(word)
        idx += 1
    return result

def gen_bars_result(data, color, orient):
    """
    Create bars
    """
    bars = []
    for label, label_df in data.groupby(color):
        if orient == 'h':
            label_df = label_df.sort_values(by='x', ascending=True)
        if label == 'a':
            label = 'lightgray'
        bars.append(go.Bar(x=label_df.x,
                           y=label_df.y,
                           name=label,
                           marker={'color': label},
                           orientation = orient,
                           text=label_df.x.astype(float).round(3),
                           hoverinfo='none',
                           textposition='auto',
                           textfont=dict(size=12, color= '#444444')
                          )
                   )
    return bars

def plot_result (qnumber):
    """
    Plot coefficients for a given question number
    """
    data = features[features.question_number == qnumber]
    title = qnumber + '. ' + data.question.values[0]
    title = split_string(title, 8)
    barplot = data[['answer', 'coefficient']].copy()
    barplot.answer = barplot.answer.apply(lambda x: split_string(x, 5))
    barplot.columns = ['y', 'x']
    bartplot = barplot.sort_values(by='x', ascending=False)
    barplot['model_highlight'] = barplot.x > 0
    barplot['color'] = barplot.model_highlight.apply(lambda x: 'cornflowerblue' if x else 'a')

    layout = gen_layout('<b>{}</b>'.format(title), 
                        'Model Coefficient', 
                        '',
                        lmarg=300,
                        h= 600)

    fig = go.Figure(data=gen_bars_result(barplot, 'color', orient='h'), 
                    layout=layout)
    iplot(fig, filename='color-bar')
plot_result('Q1')
plot_result('Q2')
plot_result('Q3')
plot_result('Q4')
plot_result('Q6')
plot_result('Q7')
plot_result('Q8')
plot_result('Q10')
plot_result('Q11')
plot_result('Q15')
plot_result('Q16')
plot_result('Q17')
plot_result('Q18')
plot_result('Q19')
plot_result('Q21')
plot_result('Q23')
plot_result('Q24')
plot_result('Q26')
plot_result('Q29')
plot_result('Q31')
plot_result('Q36')
plot_result('Q38')
plot_result('Q40')
plot_result('Q42')
plot_result('Q47')
plot_result('Q48')
plot_result('Q49')
### Training the model again with fewer questions
# Selecting just the questions we are putting in production
selected_questions = [1, 2, 3, 4, 6, 7, 8, 10, 11, 15, 16, 23, 31, 42]
treated_data = {}

# Let's select answers that had more than 5% of answers
for sq in selected_questions:
    treated_data['Q{}'.format(sq)] = treat_data(answers['Q{}'.format(sq)], sq, 0.05)
    
# Done! Now we are able to rebuild a much cleaner dataset!

# Define target variable
compensation = mcA.Q9.str.replace(',', '').str.replace('500000\+', '500-500000').str.split('-')
mcA['yearly_compensation_numerical'] = compensation.apply(lambda x: (int(x[0]) * 1000 + int(x[1]))/ 2) / 1000 # it is calculated in thousand dollars
clean_dataset = (mcA.yearly_compensation_numerical > 100).reset_index().astype(int)
clean_dataset.columns = ['index', 'top20']

# Join wit treated questions
for key, value in treated_data.items():
    value = value.reset_index(drop=True)
    clean_dataset = clean_dataset.join(value, how='left')

clean_dataset = clean_dataset.drop('index', axis=1)
# saving back to csv so others may use it
clean_dataset.to_csv('production_clean_dataset.csv')
# Create correlation matrix
correl = clean_dataset.corr().abs()

# Select upper triangle of correlation matrix
upper = correl.where(np.triu(np.ones(correl.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.5
to_drop = [column for column in upper.columns if any(upper[column] > 0.5)]

# Drop features 
clean_dataset_dropped = clean_dataset.drop(to_drop, axis=1)
# splitting train and test data
train, test = train_test_split(clean_dataset_dropped, test_size=0.2, random_state=42)
ytrain = train['top20'].copy()
Xtrain = train.drop(['top20'], axis=1).copy() # removing both target variables from features

ytest = test['top20'].copy()
Xtest = test.drop(['top20'], axis=1).copy() # removing both target variables from features

# undersampling
X_resampled, y_resampled = rus.fit_resample(Xtrain, ytrain)
# fitting the model
lreg = LogisticRegression(solver='liblinear', random_state=42)
lreg.fit(X_resampled, y_resampled)

# validating on test data
display_scores(lreg, Xtest, ytest)
# Calculates the probabilities of belonging to Top20% per range of score based on test data
calc_proba(lreg).to_frame()
input_json = {
    "Q1": "q1_other",
    "Q2": "q2_25_29",
    "Q3": "q3_united_",
    "Q4": "q4_other",
    "Q6": "q6_student",
    "Q7": "q7_other2",
    "Q8": "q8_2_3",
    "Q10": "q10_we_rec",
    "q11_analyz": "on",
    "q11_run_a_": "on",
    "q11_build_": "on",
    "q15_amazon": "on",
    "other": "on",
    "q16_python": "on",
    "q16_sql": "on",
    "Q23": "q23_25_to_",
    "q31_catego": "on",
    "q31_geospa": "on",
    "q31_numeri": "on",
    "q31_tabula": "on",
    "q31_text_d": "on",
    "q31_time_s": "on",
    "q42_revenu": "on"
}
import re
# treating the questions to match the input json
features = pd.DataFrame([Xtrain.columns, lreg.coef_[0]]).transpose()
features.columns = ['feature', 'coefficient']
features['answer'] = features.feature
features['answer'] = features['answer'].apply(lambda x: re.sub(r"[^a-zA-Z0-9]+", ' ', x))
features['answer'] = features['answer'].str.replace(' ', '_')
features['answer'] = features['answer'].str.lower()
features['answer'] = features['answer'].str.replace('_build_and_or_', '_')
features['answer'] = features['answer'].str.replace('_metrics_that_consider_', '_')
features['answer'] = features['answer'].str[:10]

features['question_number'] = features['answer'].str.split('_').str[0]
features = features[['question_number', 'answer', 'coefficient']]
features.head(3)
# treating the input json to keep it in the same format as the coeficcients
def treat_input(input_json):
    treated = dict()
    for key, value in input_json.items():
        if key[0] == 'Q':
            treated[value] = 1
        else:
            treated[key] = 1
    return treated

treated_input_json = treat_input(input_json)
print('First 8 elements of the treated input:', dict(list(treated_input_json.items())[0:8]))
features['positive'] = features['answer'].map(treated_input_json)
features.fillna(0, inplace=True)
features['points'] = features.positive * features.coefficient
features.head(5)
from math import exp
# Creating a function to normalize the scores between 0 and 1000

def normalize(points):
    """
    Normalize to get values between 0 and 1000
    """
    return int(1 / (1 + exp(-points)) * 1000)

# suming all points + intercept then normalizing between 0 and 1
score = features['points'].sum() + lreg.intercept_[0]
print('Calculated score is:', normalize(score))
import requests
import json

input_json = {
    "Q1": "q1_other",
    "Q2": "q2_25_29",
    "Q3": "q3_united_",
    "Q4": "q4_other",
    "Q6": "q6_student",
    "Q7": "q7_other2",
    "Q8": "q8_2_3",
    "Q10": "q10_we_rec",
    "q11_analyz": "on",
    "q11_run_a_": "on",
    "q11_build_": "on",
    "q15_amazon": "on",
    "other": "on",
    "q16_python": "on",
    "q16_sql": "on",
    "Q23": "q23_25_to_",
    "q31_catego": "on",
    "q31_geospa": "on",
    "q31_numeri": "on",
    "q31_tabula": "on",
    "q31_text_d": "on",
    "q31_time_s": "on",
    "q42_revenu": "on"
}

treated_input_json = treat_input(input_json)
header = {'Content-Type': 'application/x-www-form-urlencoded'}

url = 'https://tk9k0fkvyj.execute-api.us-east-2.amazonaws.com/default/top20-predictor'

requests.post(url, params=treated_input_json, headers=header).json()
# Making a get to our API. It triggers a lambda function that counts the number of objects inside our bucket.
url = 'https://wucg3iz2r4.execute-api.us-east-2.amazonaws.com/default/count-kaggle-top20-objects'
requests.get(url).json()