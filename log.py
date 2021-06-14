'''
@Author: your name
@Date: 2020-03-24 21:39:29
@LastEditTime: 2020-03-25 09:12:16
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /icd2017_final/log.py
'''

import logging
import os

'''
@description: 返回logger可保存log到文件同时打印到console上
@param {type} 
@return: 
'''
class Logger():
    def __init__(self,output_path,log_name):
        self.data_format = "%(asctime)s - %(levelname)s - %(message)s"
        self.output_path = output_path
        self.log_name = log_name

        self.init()

    def init(self):
        logger = logging.getLogger(self.log_name)
        logger.setLevel(level=logging.INFO)

        formatter = logging.Formatter(self.data_format)

        file_handler = logging.FileHandler(os.path.join(self.output_path,'{}.log'.format(self.log_name)))
        file_handler.setLevel(level=logging.INFO)
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        
        self.logger = logger

