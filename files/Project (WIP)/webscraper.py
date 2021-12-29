#This file scrapes forum posts from the official FFXIV forum. Tentatively only the healer forums. 
# The goal is to make a dashboard/website where any posts/threads on the forum can be scraped and analysed by using a link.

from sqlite3.dbapi2 import Error
from bs4 import BeautifulSoup #webparser library
import requests #web site requesting library
import re   #regular expression library to find stuff in strings
from save_data import db_manager
import datetime
from datetime import datetime as dt, timedelta
import pandas as pd
import numpy as np

db = db_manager()

print("Testing access to database...")
db.test()
db.del_test()
print('Proceeding to main task...')

main_site = 'https://forum.square-enix.com/ffxiv/'
section = 'forums/659-Healer-Roles'
url = main_site + section

response = requests.get(url)
data = response.text

soup = BeautifulSoup(data, 'html.parser')

#find threads
postlist = soup.find('ol', {'class':'threads'})

class_titles = [str(job) for job in np.ravel(pd.read_csv('jobtitles.csv', sep=',', encoding='UTF-8').values) if not str(job) == 'nan']
class_titles_string = '|'.join([str(job) for job in class_titles])
for post in postlist:
    title = ''
    link = ''
    author = ''
    first_date = ''
    first_time = ''

    if post == ' ' or post == '\n':
        continue

    try:
        title = post.find('a', {'class':'title'}).text
        link = post.find('a', {'class':'title'}).get('href')
        
        started = post.find('span', {'class':'label'})#string/tag containing name and time of creation
        author = started.find('a', {'class':'username understate'}).get('href')

        #Regular expressions for finding date and time of day
        first_date = re.findall('([0-9]{2}[-][0-9]{2}[-][0-9]{4})', 
                            started.find('a', {'class':'username understate'}).get('title'))[0]
        first_time = re.findall('([0-9]{2}[:][0-9]{2}[ ][A|P][M])', 
                            started.find('a', {'class':'username understate'}).get('title'))[0]

    except:
        continue 
        # #print('Title: ' + title + '\n' +
        # 'Link: ' + link + '\n'
        # 'Author: ' + author + '\n' +
        # 'Date: ' + first_date + '\n' +
        # 'Time: ' + first_time + '\n****\n')

    #Add or update database
    if not db.exist(title):
        try:
            db.new_table(title)
            print('Adding new table: {}'.format(title))
        except:
            break

        
    else:
        print('Table \'{}\' already exists'.format(title))
        

        #Request site
    try:
        sub_site_url = main_site + link
        new_response = requests.get(sub_site_url)
        data = new_response.text
        soup = BeautifulSoup(data, 'html.parser')
    except:
        print('Failed to connect to {}'.format(sub_site_url))
        break
    #find messages on forum board
    try:
        messageboard = soup.find('ol', {'class':'posts'})
    except:
        print('''Failed to find content:  messageboard = soup.find('ol', {'class':'posts'})''')
        break

    index = 0 #index for the message in the board
    likes = 0
    opinion = ''
    go_next_page = True

    while go_next_page == True:
        for message in messageboard:
            if message == '\n' or message == ' ' or message == '' \
            or (message.attrs['class'][0] == 'postbitdeleted' and message.attrs['class'][1] == 'postbitim'):
                continue

            try:
                #Find date and time of message post
                dtime = message.find('div', {'class':'posthead'})
                
                date = re.findall('([0-9]{2}[-][0-9]{2}[-][0-9]{4}|Yesterday|Today)', dtime.find('span', {'class':'date'}).text)[0]
                if date == 'Yesterday':
                    today = datetime.date.today()
                    yesterday = today - timedelta(days=1)
                    date = yesterday.strftime("%m-%d-%Y")
                    #print(date)
                if date == 'Today':
                    today = datetime.date.today()
                    date = today.strftime("%m-%d-%Y")
                
                time = re.findall('([0-9]{2}[:][0-9]{2}[ ][A|P][M])', dtime.find('span', {'class':'time'}).text)[0]
            except:
                print('Failed to find date and time of post')
            
            #Find author of message post
            try:
                author = message.find('a', {'class':'username offline popupctrl'}).get('href')
            except:
                author = message.find('a', {'class':'username online popupctrl'}).get('href')
            
            
            #Find from message post the text they write and remove all extra whitespaces + linebreaks
            try:
                opinion = " ".join( message.find('blockquote', {'class':'postcontent restore'}).text.split() )
            except:
                print('Opinion not found')

            try:
                likes = int(''.join( re.findall('\d' , message.find('div', {'class':'likeitic'}).text ) ) )
            except:
                likes = 0

            char_info = message.find('dl', {'class':'userinfo_mainchar'})
            # char_name = char_info.contents[2].text
            # firstname = char_name.split()[0]
            # lastname =  char_name.split()[1]
            
            #worldname = char_info.contents[5].text

            char_main_cl = re.findall('(' + class_titles_string + ')' , char_info.contents[9].text)[0]
        
            print(date)
            print(time)
            print(author)
            print(opinion)
            print('*******')
            index += 1

            #Save the data
            # To do: ..................
            mood = 'N/A'
            try:
                db.add( title, 
                        (index, 
                         author,
                         dt.strptime(date + ' ' + time, '%m-%d-%Y %I:%M %p').date(),
                         time, 
                         char_main_cl,
                         likes,
                         opinion,
                         mood) )
            except Error as e:
                print('Failed to add to table: {}'.format(title))
                print(e)

        #end for

        
        #Look for the link to the next page. Exit while-loop if no next page
        try:
            next_subpage_link = soup.find('a', {'rel':'next'}).get('href')
        except:
            next_subpage_link = ''

        if next_subpage_link:
            sub_site_url = main_site + next_subpage_link
            new_response = requests.get(sub_site_url)
            data = new_response.text
            soup = BeautifulSoup(data, 'html.parser')
            messageboard = soup.find('ol', {'class':'posts'})

        else:
            go_next_page = False
            index = 0
            
        
        
    #end while loop for iterating through all pages of posting

#end for main loop
db.close()