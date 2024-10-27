# ObsidianBot
A collection of python tools to work with Obsidian. Built with python 3.12
@Author: Sam Cullen

## goodreads_extract.py
This script will take the the goodreads export CSV and create a markdown file for each book marked as 'read' in the export file.
The md files will contain the title, book ID, date read in Obsidian internal link format, the text of your review, 
and your rating vs. the average rating of other users. 

The script will only create the file if one does not already exist

Goodreads no longer supports new API keys, so go to this link to download the CSV export file of your reviews
https://www.goodreads.com/review/import_

