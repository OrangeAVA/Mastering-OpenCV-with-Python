from pyzbar.pyzbar import decode
import cv2

img = cv2.imread("book.jpg")

decoded_objects = decode(img)

for obj in decoded_objects:
    barcode_data = obj.data.decode('utf-8')
    barcode_type = obj.type
    print(f"Barcode Type: {barcode_type}, Data: {barcode_data}")


# Display the frame
cv2.imshow('Barcode Scanner', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

import requests

def get_book_title(isbn):
    url = f'https://openlibrary.org/api/books?bibkeys=ISBN:{isbn}&jscmd=data&format=json'
    response = requests.get(url)
    
    if response.status_code == 200:
        book_data = response.json().get(f'ISBN:{isbn}')
        if book_data:
            return book_data.get('title')
    
    return "Book not found"

def get_book_info(isbn):
    url = f'https://openlibrary.org/api/books?bibkeys=ISBN:{isbn}&jscmd=data&format=json'
    response = requests.get(url)
    
    if response.status_code == 200:
        book_data = response.json().get(f'ISBN:{isbn}')
        if book_data:
            title = book_data.get('title', 'Title not found')
            
            # Retrieve author information
            authors = book_data.get('authors', [{'name': 'Author not found'}])
            author_names = [author['name'] for author in authors]
            
            # Retrieve publication information
            publish_date = book_data.get('publish_date', 'Publication date not found')
            publisher = book_data.get('publishers', [{'name': 'Publisher not found'}])
            
            return {
                'Title': title,
                'Authors': author_names,
                'Publish Date': publish_date,
                'Publisher': publisher[0]['name']
            }
    
    return {
        'Title': 'Book not found',
        'Authors': ['Author not found'],
        'Publish Date': 'Publication date not found',
        'Publisher': 'Publisher not found'
    }

def main():
    barcode_data = "9789390085361"  # Replace with your actual barcode data

    book_info = get_book_info(barcode_data)
    print(f"Book Title: {book_info}")

if __name__ == "__main__":
    main()


import pandas as pd

# Function to store book details in a CSV file using Pandas
def store_to_csv(book_info, isbn):
    # Check if the CSV file already exists, if not, create it
    try:
        df = pd.read_csv('books.csv')
    except FileNotFoundError:
        df = pd.DataFrame(columns=['ISBN', 'Title', 'Authors', 'Publish Date', 'Publisher'])
    
    # Add the 'ISBN' key to the book_info dictionary using the provided ISBN
    book_info['ISBN'] = isbn
    
    # Append the book information to the DataFrame
    df = pd.concat([df, pd.DataFrame([book_info])], ignore_index=True)
    
    # Save the DataFrame to a CSV file
    df.to_csv('books.csv', index=False)

barcode_data = '9789387944862'

book_info = get_book_info(barcode_data)
    
    # Store book details in the CSV file using Pandas
store_to_csv(book_info, barcode_data)


print(f"Book Title: {book_info['Title']}")

