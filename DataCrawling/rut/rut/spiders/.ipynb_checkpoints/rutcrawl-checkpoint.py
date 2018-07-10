# -*- coding: utf-8 -*-
import scrapy
import pandas as pd
from scrapy.http import Request
from scrapy.http import FormRequest
import csv
import os

class RutcrawlSpider(scrapy.Spider):
	name = 'rutcrawl'
	allowed_domains = ['nombrerutyfirma.cl']
	start_urls = ['https://nombrerutyfirma.cl/rut']
	data_dir = os.environ['AITUEDATA'] + "ruts_validos_scrap.csv"


	def parse(self, response):
		# return [FormRequest(url='https://nombrerutyfirma.cl/rut', formdata={"term": "18.144.865-2"}, callback=self.after_parse)]
		
		listarut = pd.read_csv(self.data_dir,header=None,names=['0','Rut']).drop('0',axis=1)

		for rut in listarut['Rut']:
			self.log("{}".format(rut))
			yield FormRequest(url='https://nombrerutyfirma.cl/rut',
			        	formdata={"term": rut}, 
			        	callback=self.after_parse,
			        	dont_filter = True)


	def after_parse(self, response):
		
		name = response.xpath('//td')[0].extract()
		name = name.replace('<td>',' ')
		name = name.replace('</td>',' ')
		rut = response.xpath('//td')[1].extract()
		rut = rut.replace('<td>',' ')
		rut = rut.replace('</td>',' ')
		gender = response.xpath('//td')[2].extract()
		gender = gender.replace('<td>',' ')
		gender = gender.replace('</td>',' ')
		adress = response.xpath('//td')[3].extract()
		adress = adress.replace('<td>',' ')
		adress = adress.replace('</td>',' ')
		comuna = response.xpath('//td')[4].extract()
		comuna = comuna.replace('<td>',' ')
		comuna = comuna.replace('</td>',' ')
		self.log("{}".format(rut))
		yield {'Name': name,
		'RUT': rut,
		'Gender': gender,
		'Adress': adress,
		'Comuna': comuna} 



