import sqlite3 as sq
from sqlite3.dbapi2 import Error
import random

# Tables with:
# forum post tables: post_id INTEGER NOT NULL, name STRING NOT NULL, datetime DATE NOT NULL, PRIMARY KEY (post_id, name, datetime), FOREIGN KEY (name) REFERENCES characters (char_id), likes INTEGER, opinion STRING NOT NULL, mood STRING


def scrub(table_name): 
    '''Scrubs the string against potential code injection'''
    s = '_'.join(table_name.split())
    s = ''.join( c for c in s if c.isalnum() or c == '_' )
    return s


class db_manager:
    '''
    Class for managing the SQL database stuff (after webscraping)
    '''

    def __init__(self, db_name = 'ffxiv_forum_data.db') -> None:
        '''
        Initialize the database.\n
        db_name: Name of the file containing the database. Alternatively use :memory: 
                if there is no file.
        '''
        self.db_name_ = db_name
        self.db_connection_ = sq.connect(self.db_name_, detect_types=sq.PARSE_DECLTYPES|sq.PARSE_COLNAMES)
        self.cursor = self.db_connection_.cursor()
    
    def test(self, testvalue = random.randint(0, 10) ):
        self.cursor.execute('CREATE TABLE IF NOT EXISTS test(t INTEGER NOT NULL)')
        self.db_connection_.commit()
        self.cursor.execute('INSERT INTO test VALUES({})'.format(testvalue))
        self.db_connection_.commit()
        print(self.cursor.execute('SELECT t FROM test').fetchall())
        print(self.cursor.execute('SELECT COUNT(t) FROM test').fetchall())

    def del_test(self):
        print('Rows in test table before deletion: {}'.format(self.cursor.rowcount))
        self.cursor.execute('DELETE FROM test;')
        self.db_connection_.commit()
        print('Rows in test table after deletion: {}'.format(self.cursor.rowcount))


    def num_changes(self):
        return self.db_connection_.total_changes()

    def close(self):
        '''
        Close the connection to the database
        '''
        
        try:
            self.db_connection_.close()

        except:
            print('Connection already closed for {name}...'.format(name=self.db_name_))


    def new_table(self, tab_name, col_info='''i INTEGER NOT NULL PRIMARY KEY, name STRING NOT NULL, datetime DATE NOT NULL, time STRING NOT NULL, likes INTEGER, main_class STRING, opinion STRING, mood STRING'''):
        '''
        Create a new table in the database
        '''
        secure_tab_name = scrub(tab_name)
        
        try:
            query = 'CREATE TABLE IF NOT EXISTS ' + secure_tab_name + '(' + col_info + ')'
            self.cursor.execute( query )
            self.db_connection_.commit()
        
        except Error as e:
            print('Error {} \nwhen creating new table: {}'.format(e, tab_name))

    
    def add(self, into_table, entry):
        '''
        into_table: name of table to insert values into\n
        entry: the values to be inserted as a [(colname, value)]
        '''
        #Loop through number of entries and add question mark ? wildcard for each coloumn for dynamic entry
        qm = ''
        max_i = len(entry)
        for i in range(max_i):
            if i < max_i - 1:
                qm = qm + '?, '
            else:
                qm = qm + '?'

        secure_tab_name = scrub(into_table)
        try:
            self.cursor.execute('INSERT INTO '+ secure_tab_name +' VALUES (' + qm + ')', entry)
        except Error as e:
            print(e)

        self.db_connection_.commit()
    

    def peek(self, table_name, coloumns='i, datetime [DATE], time, main_class, likes, opinion' , orderby='i', condition='', groupby=''):
        '''
        Peek at and return the values in a table from the database.\n
        table_name: Name of table to look at.\n
        coloumns: String in SQLite3 style the name of the coloumns to look at. What follows after ORDER BY\n
        orderby: String in SQLite3 style the coloumns to order in. What follows after WHERE.\n
        groupby: String in SQLite3 style. What follows after GROUP BY\n
        '''
        secure_tab_name = scrub(table_name)
        assert not table_name == ''

        query = 'select ' + coloumns + ' from ' + secure_tab_name + ' '
        
        if not condition == '':
            query = query + ' where ' + condition + ' '

        query = query + orderby + ' '
        
        if not groupby == '':
            query = query + ' group by ' + groupby + ' '

        result = [row for row in self.cursor.execute(query).fetchall() ]

        return result

    def list_table_names(self):
        '''
        Returns a list of the table names in the database.
        '''
        return [ str(name[0]) for name in self.cursor.execute('SELECT name FROM sqlite_master WHERE type =\'table\' AND name NOT LIKE \'sqlite_%\'').fetchall() ]
    
    def exist(self, tab_name):
        '''
        Returns whether a table name exists in the database.
        '''
        secure_tab_name = scrub(tab_name)
        result = False
        try:
            result = not self.cursor.execute('SELECT name FROM sqlite_master WHERE type=\'table\' AND name= \'?\'', secure_tab_name) == 0
        except:
            result = False

        return result