#!/usr/bin/env python
# coding: utf-8

# In[56]:


import re
class Ordinals:
    def __init__(self):
        self.text=""
    def extend_ordinal(self,text):
        ordinal_for_male={r"1ይ":r"ቀዳማይ",r"2ይ":r"ካልኣይ",r"3ይ":r"ሳልሳይ",r"4ይ":r"ራብዓይ",r"5ይ":r"ሓምሻይ",r"6ይ":r"ሻድሻይ",r"7ይ":r"ሻብዓይ",r"8ይ":r"ሻምናይ",r"9ይ":r"ታሽዓይ",r"10ይ":"ዓስራይ"}
        ordinal_for_female={r"1ይቲ":r"ቀዳመይቲ",r"2ይቲ":r"ካልእይቲ",r"3ይቲ":r"ሳልሰይቲ",r"4ይቲ":r"ራብዕይቲ",r"5ይቲ":r"ሓምሸይቲ",r"6ይቲ":r"ሻድሸይቲ",r"7ይቲ":r"ሻብዕይቲ",r"8ይቲ":r"ሻምነይቲ",r"9ይቲ":r"ታሽዕይቲ",r"10ይቲ":r"ዓስረይቲ"}
        p=re.compile("\\n")
        text=p.sub(" \\n",text)
        for i in ordinal_for_male.keys():
            p=re.compile(i)
            j=p.findall(text)
            #(j)
            text=p.sub(ordinal_for_male[i],text)
        for i in ordinal_for_female.keys():
            p=re.compile(i)
            j=p.findall(text)
            #(j)
            text=p.sub(ordinal_for_female[i],text)
        return text
    def get_text(self):
        return text


