#!/usr/bin/env python3
#==================================================================
#    created at Oct21 2019 11:29 AM 
#    author:zhaobin8
#    version:1.0
#    description:
#    usage:
#==================================================================

from distutils.core import setup
# import requests
import os

# 将markdown格式转换为rst格式
# def md_to_rst(from_file, to_file):
#       r = requests.post(url='http://c.docverter.com/convert',
#                         data={'to':'rst','from':'markdown'},
#                         files={'input_files[]':open(from_file,'rb')})
#       if r.ok:
#             with open(to_file, "wb") as f:
#                   f.write(r.content)
# md_to_rst("../../../README.md", "README.rst")

if os.path.exists('README.rst'):
      long_description = open('README.rst', encoding="utf-8").read()
else:
      long_description = 'Add a fallback short description here'


setup(name="PyXGBoost",
      version="1.0.9",
      author="zhaobin",
      author_email="303844828@qq.com",
      long_description=long_description,
      long_description_content_type="text/x-rst",
      description="pyXgboost,github:https://github.com/303844828/PyXGBoost.git",
      py_modules=["PyXGBoost"])