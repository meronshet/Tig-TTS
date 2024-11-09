#!/usr/bin/env python
# coding: utf-8

# In[3]:


from textpreprocessor import numbers_
from textpreprocessor import date_time
from textpreprocessor import cardinals
from textpreprocessor import ordinals
from textpreprocessor import acronyms
from textpreprocessor import Split_haregs
from textpreprocessor import cleaners
import re
class Normalizer:
    def __init__(self):
        self.unNormalized=""
        self.Normalized=""
    def remove_english(self,text):
        p=re.compile("[\(]{0,1}[\'|\’|\‘]{0,2}[\"]{0,1}[a-zA-Z]+[0-9]*[\s]*[\"]{0,1}[\'|\’|\‘]{0,2}[\)]{0,1}")
        text=p.sub(" ",text)
        return text
    #ኣዝማዲ ክንዲስም ብኣሕጽሮተ ቓል ዝተጻሕፈ አንተ ኣሎ ብምድላይ ይዝርግሕ
    def expand_azamadi_kndsm(self,text):
        azamadi_kndsm_dict={r"['|’]{1}ዚኣ ":r" እዚኣ ",r"['|’]{1}ዚ ":r" እዚ ",r"['|’]{1}ቲ ":r" እቲ ",r"['|’]{1}ዛ ":r" እዛ ",r"['|’]{1}ታ ":r" እታ ",                            r"['|’]{1}ዞም ":r" እዞም ",r"['|’]{1}ቶም ":r" እቶም ",r"['|’]{1}ተን ":r" እተን",r"['|’]{1}ዘን ":r" እዘን ",                            r"ዲ['|’]{1}ኻ ":r"ዶ ኢኻ ",r"ዲ['|’]{1}ዩ ":r"ዶ እዩ "}
        for i in azamadi_kndsm_dict.keys():
            p=re.compile(i)
            text=p.sub(azamadi_kndsm_dict[i],text)
        return text
    #ኣመልካቲ ክንዲስም ብኣሕጽሮተ ቓል ዝተጻሕፈ አንተ ኣሎ ብምድላይ ይዝርግሕ
    def expand_amelkati_kndsm(self,text):
        amelkati_kndsm_dict={r"['|’]{1}ዚኣተን ":r" እዚኣተን ",r"['|’]{1}ዚኣቶም ":r" እዚኣቶም ",r"['|’]{1}ቲኣቶም ":r" እቲኣቶም "                             ,r"['|’]{1}ቲኣተን ":r" እቲኣተን "}
        for i in amelkati_kndsm_dict.keys():
            p=re.compile(i)
            text=p.sub(amelkati_kndsm_dict[i],text)
        
        return text
    #ግላዊ ክንዲስም ብኣሕጽሮተ ቃል ዝተጻሕፈ አንተ ኣሎ ብምድላይ ይዝርግሕ
    def expand_glawi_kndsm(self,text):
        glawi_kndsm_dict={r"['|’]{1}የ ":r" እየ ",r"['|’]{1}ኻ ":r" ኢኻ ",r"['|’]{1}ኺ ":r" ኢኺ ",r"['|’]{1}ኹም ":r" ኢኹም ",                          r"['|’]{1}ኽን ":r" ኢኽን ",r"['|’]{1}ና ":r" ኢና ",r"['|’]{1}ዩ ":r" እዩ ",                          r"['|’]{1}ዮም ":r" እዮም ",r"['|’]{1}ያ ":r" እያ ",r"['|’]{1}የን ":r" እየን ",r"['|’]{1}ዚኣቶም ":r" እዚኣቶም "}
        for i in glawi_kndsm_dict.keys():
            p=re.compile(i)
            text=p.sub(glawi_kndsm_dict[i],text)
        
        return text
    #መስተጻምር ብኣሕጽሮተ ቃል ዝተጻሕፈ ንተ ኣሎ ብምድላይ ይዝርግሕ
    def expand_mestexamr(self,text):
        mestexamr_dict={r"['|’]{1}ውን ":r" እውን ",r"['|’]{1}ዉን ":r" እውን ",r"['|’]{1}ኣ ":r" ከኣ ",r"['|’]{1}ማ ":r" ድማ ",r"ኣሽንኳይ['|’]{1}ዶ['|’]{0,1} ":r"ኣሽንኳይዶ ",r"['|’]{1}ምበር ":r" እምበር ",r"['|’]{1}ሞ ":r" እሞ ",                        r"['|’]{1}ኳ ":r" እኳ "}
        for i in mestexamr_dict.keys():
            p=re.compile(i)
            text=p.sub(mestexamr_dict[i],text)
        return text
    def expand_informal_ahxrote_qal(self,text,choice=1):
        #,r"እንተ['|’]{1} ":r"እንተ ኣ"
        informal_ahxroteqal_dict_1={r"ን['|’]{1}":r"ን",r"እንዳ['|’]{1}ባ ":r" እንዳባ ",r"['|’]{1}ለኻ ":r" ኣለኻ ",r"['|’]{1}ዝን ":r" እዝን ",                                    r"['|’]{1}ትን ":r" እትን ",r"['|’]{1}ዃ ":r" እኳ",r"['|’]{1}ድኣ ":r" ድኣ ",r"መን['|’]{1}":r"መን ",                                    r"['|’]{1}ቦይ ":r" ኣቦይ ",r"['|’]{1}ቦኣ ":r" ኣቦኣ ",r"['|’]{1}መን ":r" መን "}
        informal_ahxroteqal_dict_2={r" ስ['|’]{1}ር ":r" ስዉር ",r" እንከለ['|’]{1} ":r" እንከለዉ ",r" ብዘለ['|’]{1} ":r" ብዘለዉ ",                                    r" ዘለ['|’]{1} ":r" ዘለዉ ",r" ምህላ['|’]{1} ":r" ምህላዉ ",r"['|’]{1}ልዮም ":r" ኣልዮም ",                                    r"['|’]{1}ምተኻእለን ":r" ኣይምተኻእለን ",r" ኣለ['|’]{1} ":r" ኣለዉ ",r" ክኣት['|’]{1} ":r" ክኣትዉ ",                                    r"['|’]{1}ስተየ ":r" ኣስተየ "}
        informal_ahxroteqal_dict_3={r"['|’]{1}ተሎ ":r"ተ ኣሎ ",r"ብ['|‘]{1}":r"ብ",r"ን['|‘]{1}":r"ን",r"['|’]{1}ኸ":r"ኸ",                                    r"['|’]{1}ተኾነ ":r" እንተኾነ ",r"['|’]{1}ለና ":r" ኣለና ",r"['|’]{1}ባ ":r" እባ ",                                    r"['|’]{1}ምበኣር ":r" እምበኣር ",r"['|’]{1}ለኹ ":r" ኣለኹ ",r"['|’]{1}ን ":r"ን ",                                    r"['|’]{1}ንበብካ ":r" ኣንበብካ ",r"['|’]{1}ድነቕካን ":r" ኣድነቕካን "}
        informal_ahxroteqal_dict_4={r"['|’]{1}ንስትዮ ":r" ኣንስትዮ ",r"['|’]{1}ላ ":r" ኣላ ",                                    r"['|’]{1}ለዋ ":r" ኣለዋ ",r"['|’]{1}ወ ":r" እወ ",                                    r" ዘተኣታተ['|’]{1} ":r" ዘተኣታተዉ ",r" ክሕል['|’]{1} ":r" ክሕልዉ ",r" እና['|’]{1} ":r" እና "}
        informal_ahxroteqal_dict_5={r"ዶ['|’]{1}ምበርከ":r"ዶ እምበርከ",r"['|’]{1}ዶ ":r"ዶ ",r"['|’]{1}ሉ ":r" ኢሉ ",r"['|’]{1}ስኪ ":r" እስኪ ",                                    r"['|’]{1}ስከ ":r" እስከ ",r"['|’]{1}ኮ ":r" እኮ ",r"['|’]{1}ስ ":r"ስ ",r"['|’]{1}ሲ ":r"ሲ ",                                    r"['|’]{1}ጥረየ ":r" ኣጥረየ ",r"['|’]{1}ለዉ ":r" ኣለዉ ",r"['|’]{1}ካ ":r" ኣለካ ",                                    r"['|’]{1}ለዎ ":r" ኣለዎ ",r"['|’]{1}ልያ ":r" ኣልያ "}
        informal_ahxroteqal_dict_6={r"['|’]{1}ንድዩ ":r" እንድዩ ",r"['|’]{1}ልገብናዮ ":r" ኣልገብናዮ ",r"['|’]{1}ጥፍአን ":r" ኣጥፍአን ",                                    r"['|’]{1}ደ ":r" ኣደ ",r"['|’]{1}ሎካ ":r" ኣሎካ ",r"['|’]{1}ሎ ":r" ኣሎ ",                                    r"['|’]{1}ጋጢምዎም ":r" ኣጋጢምዎም ",r"['|’]{1}ንቈልቈለ ":r" ኣንቈልቈለ ",r"['|’]{1}ንቆልቆለ ":r" ኣንቆልቆለ ",                                    r"['|’]{1}ንጠብና ":r" ኣንጠብና ",r"['|’]{1}ቕሪበን ":r" ኣቕሪበን ",r"['|’]{1}ንስተይቲ ":r" ኣንስተይቲ "}
        which={}
        if(choice==1):
            which=informal_ahxroteqal_dict_1
        elif(choice==2):
            which=informal_ahxroteqal_dict_2
        elif(choice==3):
            which=informal_ahxroteqal_dict_3
        elif(choice==4):
            which=informal_ahxroteqal_dict_4
        elif(choice==5):
            which=informal_ahxroteqal_dict_5
        elif(choice==6):
            which=informal_ahxroteqal_dict_6
        for i in which.keys():
            p=re.compile(i)
            text=p.sub(which[i],text)
        return text
    def remove_double_qoutes(self,text):
        p=re.compile(r"“")
        text=p.sub("",text)
        p=re.compile(r"”")
        text=p.sub("",text)
        p=re.compile(r"‘’")
        text=p.sub("",text)
        p=re.compile(r"’’")
        text=p.sub("",text)
        p=re.compile("\"")
        text=p.sub("",text)
        p=re.compile("\`")
        text=p.sub("'",text)
        p=re.compile("\?")
        text=p.sub(" ?",text)
        p=re.compile("\!")
        text=p.sub(" !",text)
        p=re.compile(r"፤")
        text=p.sub(r" ፤",text)
        p=re.compile(r"‚")
        text=p.sub(",",text)
        return text
    def restore_fused_words(self,text):
        fused={"ኣብ እዚ":"ኣብዚ","ካብ እዚ":"ካብዚ","ናብ እዚ":"ናብዚ","ናይ እዚ":"ናይዚ"}
        for i in fused.keys():
            p=re.compile(i)
            text=p.sub(fused[i],text)
        return text
    def normalize(self,text):
        text=self.remove_english(text)
        text=self.remove_double_qoutes(text)
        text=self.expand_glawi_kndsm(text)
        text=self.expand_azamadi_kndsm(text)
        text=self.expand_amelkati_kndsm(text)
        text=self.expand_mestexamr(text)
        text=self.expand_informal_ahxrote_qal(text,1)
        text=self.expand_informal_ahxrote_qal(text,2)
        text=self.expand_informal_ahxrote_qal(text,3)
        text=self.expand_informal_ahxrote_qal(text,4)
        text=self.expand_informal_ahxrote_qal(text,5)
        text=self.expand_informal_ahxrote_qal(text,6)
        norm_date_time=date_time.date_and_time()
        text=norm_date_time.normalize_date(text)
        text=norm_date_time.normalize_time(text)
        norm_cardinals=cardinals.Cardinals()
        text=norm_cardinals.normalize_cardinals(text)
        norm_ordinals=ordinals.Ordinals()
        text=norm_ordinals.extend_ordinal(text)
        acro=acronyms.Acronyms()
        text=acro.normalize_acronyms(text)
        num=numbers_.Numbers()
        text=num.normalize_fractions(text)
        text=num.normalize_telephone_number(text)
        text=num.normalize_per_line(text)
        cleaner=cleaners.Cleaners()
        text=cleaner.clean(text)
        return text


# In[4]:


#history,relationship,politics,law,social,sport,culture,health,business,agriculture
#"split domains/new-"+
if __name__=='__main__':
    print("enter the name of files to edit in comma seperated list")
    oldfilelist=input("enter old filelist:")
    files=oldfilelist.split(",")
    for i in files:
        file=open(i+".txt","r",encoding="UTF-8")
        newfile=open("split domains/new-"+i+".txt","w",encoding="UTF-8")
        text=file.read()
        #file.close()
        splitter=Split_haregs.Splitter(text)
        splitter.colon_problem()
        splitter.colon_problem()
        splitter.paragraph_tokenizer()
        splitter.average_sentences(151)
        splitter.Hareg_Sampler(3000)
        print(splitter.average_length())
        splitter.freq_dist_of_hareg_lengths()
        text=splitter.get_SampleHaregs()
        #text=splitter.get_Haregs()
        normalizer=Normalizer(text)
        text=normalizer.normalize(text)
        newfile.write(text)
        file.close()
        newfile.close()


# In[ ]:





# In[ ]:





# In[ ]:




