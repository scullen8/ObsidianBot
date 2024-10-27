"""
This script will take the the goodreads export CSV and create a markdown file for each book marked as 'read' in the export file.
The md files will contain the title, book ID, date read in Obsidian internal link format, the text of your review, 
and your rating vs. the average rating of other users. 

The script will only create the file if one does not already exist

Goodreads no longer supports new API keys, so go to this link to download the CSV export file of your reviews
https://www.goodreads.com/review/import

"""

import csv
import os
import re
import datetime


def sanitize_file_name(title, max_length=150):
    # Remove special characters not allowed in Windows file names
    sanitized_title = re.sub(r'[\/:*?"<>|]', '', title)
    
    # Remove trailing dots and spaces
    sanitized_title = sanitized_title.rstrip('. ').rstrip()
    
    # Truncate the title if it's too long
    if len(sanitized_title) > max_length:
        sanitized_title = sanitized_title[:max_length]
    
    return sanitized_title


def create_text_file(ob_book_dir, title, review):
    filename = f"{ob_book_dir}/{title}"

    with open(filename, 'w', encoding='utf-8') as file:
        file.write(review)

def main(csv_file_path, ob_book_dir):

    exisitngBooks = os.listdir(ob_book_dir)
    # Filtering only the files.
    exisitngBooks = [f for f in exisitngBooks if os.path.isfile(ob_book_dir+'/'+f)]

    #print(*exisitngBooks, sep="\n")

    with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        
        # Skip the header if present
        next(csv_reader, None)
        
        for col in csv_reader:
            if len(col) >= 20 and col[18] == 'read':
                bID = col[0]
                title = f"{sanitize_file_name(col[1])}.md"
                author = col[2]
                mRating = col[7]
                rating = col[8]
                pages = col[11]
                date = ''
                if col[14] != '':
                    date = "[["+datetime.datetime.strptime(col[14], '%Y/%m/%d').strftime('%Y-%m-%d')+"]]"
                review = col[16]
                rCount = col[19]

                content = f"Book ID: {bID}\nAuthor: {author} Read: {date}\n{review}\nMy Rating: {mRating} vs. avg: {rating}\n{pages} pages and read {rCount}\n\n"
                
                if title not in exisitngBooks:
                    print(title)
                    create_text_file(title, content)



if __name__ == "__main__":
    csv_file_path = "goodreads_library_export.csv"
    ob_book_dir = 'Books'
    
    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file not found at {csv_file_path}.")
    elif not os.path.exists(ob_book_dir):
        print(f"Error: Book review folder not found {ob_book_dir}.")
    else:
        main(csv_file_path, ob_book_dir)
        print("Text files created successfully.")