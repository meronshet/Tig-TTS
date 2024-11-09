#!/usr/bin/env python
# coding: utf-8

# In[20]:


import re
from textpreprocessor import numbers_
class Cardinals:
    def __init__(self):
         self.text=""
    def normalize_cardinals(self,text):
        num_normalizer=numbers_.Numbers()
        p=re.compile("[0-9]+ታት")
        l=p.findall(text)
        for i in l:
            p=re.compile("ታት")
            str1=p.sub("",i)
            if(str1.isdigit()):
                #(i)
                #(str1)
                form=num_normalizer.normalize_numbers(int(str1))
                #(form)
                if(form.endswith("ን")):
                    last_form=form[0:len(form)-1]+"ታት"
                else:
                    last_form=form+"ታት"
                #(last_form)
            p=re.compile(i)
            text=p.sub(last_form,text)
        return text


# In[21]:


if(__name__=='__main__'):
    newfile=open("ekkebkab numbers.txt","r",encoding="UTF-8")
    text=newfile.read()
    newfile.close()
    card=Cardinals()
    text=card.normalize_cardinals(text)
    newfile=open("ekkebkab numbers.txt","w",encoding="UTF-8")
    newfile.write(text)


# In[ ]:




