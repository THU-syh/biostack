import scrapy
from scrapy.selector import Selector

class DHISpider(scrapy.Spider):
    name = "medlinePlus"
    start_urls = [
       'https://medlineplus.gov/druginfo/herb_All.html',
    ]  

    def parse(self, response):
        
        for href in response.xpath('//div[@class="section-body"]//ul//li//a/@href').extract():
            yield response.follow(href, self.parse_name)
    
    def parse_name(self, response):
        def extract_with_css(query):
            return response.xpath(query).extract()

        yield {
            #MEDLINEPLUS
            'name_herbs': extract_with_css('normalize-space(.//article//div[@class="page-info"]//div[@class="page-title"]//h1[@class="with-also"]/text())'), 
            'descriptions': extract_with_css('//article//section//div[@id="Description"]//div[@class="section-body"]/text()'),
            'drug_interaction': extract_with_css('//article//section//div[@id="DrugInteractions"]//div[@class="section-body"]//dl//dd/text()'),
            'latin_names': extract_with_css('//article//section//div[@id="OtherNames"]//div[@id="section-other"]/text()'), 
        }