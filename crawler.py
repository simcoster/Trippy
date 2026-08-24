"""
Crawler to discover campsites from parks.org.il
Extracts campsite elements and their href/title information.

Uses `httpx` for HTTP requests and BeautifulSoup for parsing.
"""

import httpx
from bs4 import BeautifulSoup
from typing import List, Dict
import json


def crawl_campsites(url: str) -> List[Dict[str, str]]:
    """
    Crawl the campsite listing page and extract campsite information.
    
    Args:
        url: The URL to crawl
        
    Returns:
        List of dictionaries with 'href' and 'title' keys
    """
    # Set headers to mimic a browser request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    campsite_elements = []
    
    try:
        # Fetch the page
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url, headers=headers, follow_redirects=True)
            response.raise_for_status()
            
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all campsite elements
        campsite_elements = soup.find_all('div', class_=lambda c: c and 'team_repeater_wrapper' in c)
        
        print(f"Found {len(campsite_elements)} campsite elements")
        
        campsites= []
        # Extract href and title from each campsite
        for campsite in campsite_elements:
            # Look for anchor tag with class "title" within the campsite
            
            href = campsite.select_one("a")["href"]
            title = campsite.select_one("h2").get_text(strip=True)
                
                # If href is relative, make it absolute
            if href and not href.startswith('http'):
                from urllib.parse import urljoin
                href = urljoin(url, href)
            
            if href and title:
                campsites.append({
                    'href': href,
                    'title': title
                })
                print(f"Found: {title[:50]}... -> {href[:80]}...")
        return campsites
        
    except httpx.HTTPError as e:
        print(f"HTTP error occurred: {e}")
        return []


def main():
    """Main function to run the crawler."""
    url = "https://www.parks.org.il/%D7%94%D7%96%D7%9E%D7%A0%D7%95%D7%AA-%D7%9C%D7%97%D7%A0%D7%99%D7%95%D7%A0%D7%99-%D7%9C%D7%99%D7%9C%D7%94/"
    
    print(f"Crawling: {url}")
    print("-" * 80)
    
    campsites = crawl_campsites(url)
    
    print("-" * 80)
    print(f"\nTotal campsites found: {len(campsites)}")
    
    # Save to JSON file
    output_file = "campsites.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(campsites, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Print first few results as preview
    if campsites:
        print("\nFirst 5 results:")
        for i, site in enumerate(campsites[:5], 1):
            print(f"{i}. {site['title']}")
            print(f"   {site['href']}\n")


if __name__ == "__main__":
    main()
