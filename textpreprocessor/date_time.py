#!/usr/bin/env python
# coding: utf-8

# In[20]:


from textpreprocessor import numbers_
import re
class date_and_time:
    def __init__(self):
        ("")
    def normalize_time(self,text):
        p=re.compile("[0-6]?[0-9]\:[0-6]?[0-9]|[0-6]?[0-9]\:[0-6]?[0-9]\:[0-6]?[0-9]")
        l=p.findall(text)
        num=numbers_.Numbers()
        for i in l:
            time_parts=i.split(r":")
            if(len(time_parts)==3):
                hour=int(time_parts[0])
                minute=int(time_parts[1])
                seconds=int(time_parts[2])
                if(minute==0 and seconds==0):
                    expand_form=num.normalize_numbers(hour)+" ሰዓት"
                elif(minute==0):
                    expand_form=num.normalize_numbers(hour)+" ሰዓትን "+num.normalize_numbers(seconds)+" ካልኢትን"
                elif(seconds==0):
                    expand_form=num.normalize_numbers(hour)+" ሰዓትን "+num.normalize_numbers(minute)+" ደቂቅን"
                else:
                    expand_form=num.normalize_numbers(hour)+" ሰዓትን "+num.normalize_numbers(minute)+" ደቂቅን "+num.normalize_numbers(seconds)+" ካልኢትን"
            else:
                hour=int(time_parts[0])
                minute=int(time_parts[1])
                extra=minute%5
                if(extra>2):
                    minute=(minute//5)*5+5
                else:
                    minute=(minute//5)*5
                if(minute==0):
                    expand_form="ሰዓት "+num.normalize_numbers(hour)
                elif(minute>=35):
                    minute=60-minute
                    expand_form="ንሰዓት "+num.normalize_numbers(hour+1)+" "+num.normalize_numbers(minute)+" ጎደል"
                else:
                    expand_form="ሰዓት "+num.normalize_numbers(hour)+"ን "+num.normalize_numbers(minute)+"ን"
            p=re.compile(i)
            text=p.sub(expand_form,text)
        return text
    def normalize_date(self,text):
        p=re.compile("[0-9]+\/[0-9]+\/[0-9]+")
        l=p.findall(text)
        num=numbers_.Numbers()
        for i in l:
            date_parts=i.split(r"/")
            if(len(date_parts[0])>=3):
                year=int(date_parts[0])
                if(int(date_parts[1])>12):
                    day=int(date_parts[1])
                    month=int(date_parts[2])
                else:
                    day=int(date_parts[2])
                    month=int(date_parts[1])
            elif(len(date_parts[2])>=3):
                year=int(date_parts[2])
                if(int(date_parts[1])>12):
                    day=int(date_parts[1])
                    month=int(date_parts[0])
                else:
                    day=int(date_parts[0])
                    month=int(date_parts[1])
            else:
                if(int(date_parts[2])>=22):
                    year=int("19"+date_parts[2])
                else:
                    year = int("20" + date_parts[2])
                if(int(date_parts[1])>12):
                    day=int(date_parts[1])
                    month=int(date_parts[0])
                else:
                    day=int(date_parts[0])
                    month=int(date_parts[1])
            expand_form="ዕለት "+num.normalize_numbers(day)+" ወርሒ "+num.normalize_numbers(month)+" "+num.normalize_numbers(year)
            p=re.compile(i)
            text=p.sub(expand_form,text)
        return text


# In[ ]:





# In[ ]:




