import requests
import csv
import urllib.parse
import bs4
import os

class WebScraper:
    def __init__(self, start_url, output_filename="data/books.csv"):
        self.base_url = start_url
        self.current_url = start_url
        self.output_filename = output_filename
        self.books_data = []

    def fetch_page(self, url):
        try:
            response = requests.get(url,timeout=10)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None
    def parse_book_details(self,book_soup):
        try:
            title_tag=book_soup.find('h3').find('a')
            title=title_tag['title']
            book_url_relative=title_tag['href']
            current_page_url=self.current_url
            book_url=urllib.parse.urljoin(current_page_url,book_url_relative)
            price_tag=book_soup.find('p',class_ = 'price_color')
            price = price_tag.text
            
            stock_tag=book_soup.find('p',class_='instock availability') 
            stock= stock_tag.text.strip()

            
            rating_tag=book_soup.find('p',class_='star-rating')
            rating_classes=rating_tag['class'] if rating_tag and rating_tag.has_attr('class') else []
            rating=rating_classes[1] if len(rating_classes) > 1 else ''

            return{
                'title':title,
                'price':price,
                'stock':stock,
                'rating':rating,
                'url':book_url,

            }
        except (AttributeError,IndexError,KeyError) as e:
            print(f"Error parsing book_details: {e}")
            return None               
    def parse_page(self,html):
        soup=bs4.BeautifulSoup(html,'html.parser')
        books=soup.find_all('article',class_='product_pod')


        if not books:
            return None


        for book_soup in books:
            details=self.parse_book_details(book_soup)

            if details:
                self.books_data.append(details)
        return soup.find('li',class_= 'next')

    def save_to_csv(self):
        if not self.books_data:
            print("No data to save.")
            return


        keys=self.books_data[0].keys()
        
        output_dir = os.path.dirname(self.output_filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(self.output_filename,'w',newline='',encoding='utf-8') as output_file:
            dict_writer=csv.DictWriter(output_file,keys)
            dict_writer.writeheader()
            dict_writer.writerows(self.books_data)

        print(f"Data saved to {self.output_filename}")

    def run(self):
        while self.current_url:
            print(f"scraping {self.current_url}")
            html_content=self.fetch_page(self.current_url)
            if not html_content:
                break
            next_page=self.parse_page(html_content)

            if next_page and next_page.find('a'):
                next_page_relative_url=next_page.find('a')['href']
                self.current_url=urllib.parse.urljoin(self.current_url,next_page_relative_url)
            else:
                self.current_url=None
        print(f"scraping complete. {len(self.books_data)} books found.")
        self.save_to_csv()


if __name__ == "__main__":
    
    start = "http://books.toscrape.com/"
    scraper = WebScraper(start_url=start, output_filename="data/books.csv")
    scraper.run()





                                      


         

