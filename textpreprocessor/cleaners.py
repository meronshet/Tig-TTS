#!/usr/bin/env python
# coding: utf-8

# In[14]:

import re
class Cleaners:
    def __init__(self):
        ("")
    def clean(self,text):#'!\'(),.።<፣፥፡?፧፨፠፤፦ '
        p=re.compile("\.\.\.|\-|\||\]|\[|\}|\{|\>|\<|\•|\—|\/")
        text=p.sub(" ",text)
        p=re.compile("\"|\*|\%|\$|\~|\@|\#|\^|\&|\_|\=|\+|\;|\:|\’|\‘|…")
        text=p.sub("",text)
        p=re.compile("\/")
        text=p.sub(" ",text)
        p=re.compile("[ ]+")
        text=p.sub(" ",text)
        return text


# In[15]:


if __name__=='__main__':
    newfile=open("split domains/new-agriculture"+".txt","r",encoding="UTF-8")
    text=newfile.read()
    c=Cleaners()
    text=c.clean(text)
    file=open("cleaned_text.txt","w",encoding="UTF-8")
    file.write(text)


# In[ ]:




