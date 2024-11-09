#!/usr/bin/env python
# coding: utf-8

# In[78]:


import re
class Numbers:
    def __init__(self):
        self.normal_form=[]
        self.digits=[]
        self.line=0
    def once_digits(self,number):
        once_digits={0:"ባዶ",1:"ሓደ",2:"ክልተ",3:"ሰለስተ",4:"ኣርባዕተ",5:"ሓሙሽተ",6:"ሽዱሽተ",7:"ሸውዓተ",8:"ሸሞንተ",9:"ትሽዓተ"}
        return once_digits[number]
    def tenth_digits(self,number):
        tenth_digits={10:"ዓሰርተ",20:"ዒስራ",30:"ሳላሳ",40:"ኣርብዓ",50:"ሓምሳ",60:"ሱሳ",70:"ሰብዓ",80:"ሰማንያ",90:"ቴስዓ",100:"ሚእቲ"}
        return tenth_digits[number]
    def get_digits(self,number):
        digits=[]
        number1=number
        if(number!=0):
            while(number1>0):
                digit=number1%10
                number1=number1//10
                digits.insert(0,digit)
        else:
            digits.insert(0,number)
        ##(digits)
        return digits
    def normalize_fractions(self,text):
        p=re.compile("\s*\/\s*")
        text=p.sub("/",text)
        p=re.compile("[\-\+]?[0-9][\,0-9]* [\-\+]?[0-9][\,0-9]*[\/][0-9][\,0-9]*")
        l=p.findall(text)
        for i in l:
            mlue_quxri=i.split(" ")
            fraction=mlue_quxri[1].split("/")
            norm_form=self.normalize_real_numbers(mlue_quxri[0])+" ምሉእን "+self.normalize_real_numbers(fraction[0])+" ኣብ ልዕሊ "+            self.normalize_real_numbers(fraction[1])+"ን"
            p=re.compile(i)
            text=p.sub(norm_form,text)
        return text
    def normalize_per_line(self,text):
        lines=text.split("\n")
        normalized=""
        j=0
        i=0
        while(i<len(lines)):
            j+=1
            self.line=j
            if(i!=len(lines)-1):
                normalized=normalized+self.filter_and_normalize(lines[i])+"\n"
            else:
                normalized=normalized+self.filter_and_normalize(lines[i])

            i+=1
        return normalized
    def filter_and_normalize(self,text):
        text=str(self.normalize_fractions(text))
        p=re.compile("\s*-\s*")
        text=p.sub("-",text)
        p=re.compile("\s*\/\s*")
        text=p.sub("/",text)
        #p=re.compile("[0-9][0-9]*[.]*[0-9]*[\-0-9]?[0-9]*[.]*[0-9]*")
        p=re.compile("[\-\+]?[0-9][\,]?[0-9]{0,3}[\,]?[0-9]{0,3}[\,]?[0-9]{0,3}[\,]?[0-9]{0,3}[\.]{0,1}[0-9]*[\%]?[\/\-\—]{0,1}[\-\+]?[0-9]{0,1}[\,]?[0-9]{0,3}[\,]?[0-9]{0,3}[\,]?[0-9]{0,3}[\,]?[0-9]{0,3}[\.]{0,1}[0-9]*[\%]?")
        l=p.findall(text)
        t=l
        if(l!=None):
            iterate=len(l)
            cursor=0
            while(cursor<iterate-1):
                cursor1=cursor+1
                while(cursor1<iterate):
                    x=t[cursor]
                    if(len(x)>len(t[cursor1])):
                        temp=t[cursor1]
                        t[cursor1]=x
                        t[cursor]=temp
                    cursor1+=1
                cursor+=1
        else:
            ierate=0
        while(iterate>0):
            iterate=iterate-1
            i=t[iterate]
            ##(i)
            j=i
            i=i.strip()
            i=i.replace(",","")
            i=i.replace("—","-")
            if(i.find("/")!=-1):
                ##(i)
                range_=i.split("/")
                ##(range_)
                if(i.endswith("/")==False):
                    if(text.find("ኣዋጅ")!=-1 or text.find("ሕጊ")!=-1 or text.find("ስርዓት")!=-1):
                        norm_form=self.normalize_real_numbers(range_[0])+" እዝባር "+self.normalize_real_numbers(range_[1])
                    elif(len(range_[0])>len(range_[1])and len(range_[0])==4 and len(range_[1])==2):
                        temp_=range_[0]
                        str_new=temp_[:int(len(temp_)/2)]+range_[1]
                        if(int(range_[0])+1==int(str_new)):
                            norm_form=self.normalize_real_numbers(range_[0])+" "+self.normalize_real_numbers(str_new)
                        else:
                            norm_form=self.normalize_real_numbers(range_[0])+" ክሳብ "+self.normalize_real_numbers(str_new)
                    elif(len(range_[0])==4 and len(range_[1])==4):
                        if(int(range_[0])+1==int(range_[1])):
                            norm_form=self.normalize_real_numbers(range_[0])+" "+self.normalize_real_numbers(range_[1])
                        else:
                            norm_form=self.normalize_real_numbers(range_[0])+" ክሳብ "+self.normalize_real_numbers(range_[1])
                    else:
                        norm_form=self.normalize_real_numbers(range_[0])+" ኣብ ልዕሊ "+self.normalize_real_numbers(range_[1])
                else:
                    continue
            elif(i.find("-",1)!=-1 and i.find("-",1)!=len(i)-1):
                hyphen_position=i.find("-",1)
                range_=[i[0:hyphen_position],i[hyphen_position+1:]]
                if(len(range_[0])<=len(range_[1])):
                    norm_form=self.normalize_real_numbers(range_[0])+" ክሳብ "+self.normalize_real_numbers(range_[1])
                else:
                    temp_=range_[0]
                    str_new=temp_[:int(len(temp_)/2)]+range_[1]
                    norm_form=self.normalize_real_numbers(range_[0])+" ክሳብ "+self.normalize_real_numbers(str_new)
            else:
                norm_form=self.normalize_real_numbers(i)
            p=re.compile(j)
            text=p.sub(norm_form,text)
        return text
    def normalize_real_numbers(self,text):
        text=text.strip()
        p=re.compile("\.0+$")
        text=p.sub("",text)
        if(text.startswith("0") and text.find(".")==-1 and text!="0"):
            expand_form=self.normalize_serial_numbers(text)
        else:
            if(text.find('%')!=-1):
                expand_form=self.normalize_real_numbers(text.replace("%",""))+" ሚእታዊት"
            elif(text.find(".")!=-1):
                if(text.find(".")!=len(text)-1):
                    #truncating trailing zeros
                    p=re.compile("0+$")
                    text=p.sub("",text)
                    decimal_point=text.split(".")
                    point_expand=""
                    for i in decimal_point[1].strip():
                        point_expand=point_expand+" "+self.normalize_numbers(int(i))
                    if(decimal_point[0].find("-")!=-1):
                        expand_form="ኣሉታ "+self.normalize_numbers(int(decimal_point[0].replace("-","")))+" ነጥቢ "+point_expand.strip()
                    elif(decimal_point[0].find("+")!=-1):
                        expand_form="ኣወንታ "+self.normalize_numbers(int(decimal_point[0].replace("+","")))+" ነጥቢ "+point_expand.strip()
                    else:
                        expand_form=self.normalize_numbers(int(decimal_point[0]))+" ነጥቢ "+point_expand.strip()
                else:
                    expand_form=self.normalize_numbers(int(text.replace(".","")))+"."
            else:
                if(text.find("-")!=-1):
                    expand_form="ኣሉታ "+self.normalize_numbers(int(text.replace("-","")))
                elif(text.find("+")!=-1):
                    expand_form="ኣወንታ "+self.normalize_numbers(int(text.replace("+","")))
                else:
                    expand_form=self.normalize_numbers(int(text))
        return expand_form
    def normalize_serial_numbers(self,text):
        i=0
        expand_form=""
        while(i<len(text)-1):
            if(i==len(text)-3 and len(text)%2!=0):
                if(text[i]=="0"):
                    expand_form=expand_form+" "+self.normalize_numbers(int(text[i]))+" "+self.normalize_numbers(int(text[i+1]))
                else:
                    expand_form=expand_form+" "+self.normalize_numbers(int(text[i:i+2]))
                expand_form=expand_form+" "+self.normalize_numbers(int(text[i+2]))
                break
            else:
                if(text[i]=="0"):
                    expand_form=expand_form+" "+self.normalize_numbers(int(text[i]))+" "+self.normalize_numbers(int(text[i+1]))
                else:
                    expand_form=expand_form+" "+self.normalize_numbers(int(text[i:i+2]))
            i+=2
        return expand_form
    def normalize_telephone_number(self,text):
        p=re.compile("[\+]?291?[\-\s]{0,1}07[0-9]{6}|[\+]?[\-\s]{0,1}07[0-9]{6}")#")
        l=p.findall(text)
        p=re.compile("(18\s[0-9]{2}\s[0-9]{2}|11\s[0-9]{2}\s[0-9]{2}|12\s[0-9]{2}\s[0-9]{2}|15\s[0-9]{2}\s[0-9]{2}|12\s[0-9]{2}\s[0-9]{2}|20\s[0-9]{2}\s[0-9]{2})")
        l2=p.findall(text)
        l=l+l2
        for i in l:
            j=i
            i=i.replace("+","")
            i=i.replace(" ","")
            i=i.replace("-","")
            expand_form="yes"
            if(i.startswith("291")):
                expand_form=self.normalize_numbers(int(i[:3]))+" "
                expand_form=expand_form+" "+self.normalize_serial_numbers(i[3:])
            else:
                expand_form=expand_form+" "+self.normalize_serial_numbers(i)
            j=j.replace("+",r"\+")
            p=re.compile(j)
            text=p.sub(expand_form,text)
            #(expand_form)
        return text
    def digits_normal_form(self,digits_list):
        i=len(digits_list)-1
        digits_reverse=[]
        while(i>=0):
            digits_reverse.append(digits_list[i])
            i=i-1
        digits_list=digits_reverse
        i=len(digits_list)-1
        normal_form=[]
        text=""
        ##(digits_reverse)
        while(i>=0):
            rem=(i+1)%3
            if(rem==1):
                if(digits_list[i]!=0):
                    text=self.once_digits(digits_list[i])
                else:
                    text=""
            elif(rem==0):
                if(digits_list[i]!=0):
                    text=self.once_digits(digits_list[i])
                else:
                    text=""
            else:
                if(digits_list[i]!=0):
                    text=self.tenth_digits(digits_list[i]*10)
                else:
                    text=""
            normal_form.append(text)
            ##(self.digits[i])
            ##(text)
            ##(rem)
            i=i-1
        return normal_form
    def normalize_tenth_numbers(self,j):
        reading_form=[]
        if(self.digits[j+1]!=0 and self.digits[j]!=1 and self.digits[j]!=0):
            text=self.normal_form[j]+"ን "+self.normal_form[j+1]+"ን"
            reading_form.append(text)
        elif(self.digits[j]==1 and self.digits[j+1]!=0):
            if(j==0):
                text=self.normal_form[j]+" "+self.normal_form[j+1]
                reading_form.append(text)
            else:
                text=self.normal_form[j]+" "+self.normal_form[j+1]+"ን"
                reading_form.append(text)
        elif(self.digits[j+1]==0):
            if(j==0):
                text=self.normal_form[j]
                reading_form.append(text)
            else:
                text=self.normal_form[j]+"ን"
                reading_form.append(text)
        else:
            text=self.normal_form[j+1]+"ን"
            reading_form.append(text)
        return "".join(reading_form)
    def normalize_hundreds_number(self,j):
        reading_form=[]
        if(self.digits[j+2]==0 and self.digits[j+1]==0):
            if(self.digits[j]==1):
                if(j+3!=len(self.digits) or j==0):
                    text="ሚእቲ"
                else:
                    text="ሚእትን"
                reading_form.append(text)
            else:
                if(j+3!=len(self.digits) or j==0):
                    text=self.normal_form[j]+" ሚእቲ"
                else:
                    text=self.normal_form[j]+" ሚእትን"
                reading_form.append(text)
            ##(reading_form)
        else:
            if(self.digits[j]!=0):
                if(self.digits[j]==1):
                    text="ሚእትን "
                    reading_form.append(text)
                elif(self.digits[j]!=1):
                    text=self.normal_form[j]+" ሚእትን "
                    reading_form.append(text)
            ##(text)
            reading_form.append(self.normalize_tenth_numbers(j+1))
            ##(reading_form)
        return "".join(reading_form)
    def normalize_numbers(self,number):
        number=abs(number)
        self.digits=self.get_digits(number)
        self.normal_form=self.digits_normal_form(self.digits)
        ##(self.digits)
        ##(self.normal_form)
        reading_form=[]
        i=len(self.digits)-1
        if(number<10):
            return self.once_digits(number)
        elif(number<100):
            reading_form.append(self.normalize_tenth_numbers(0))
        elif(number<1000):
            reading_form.append(self.normalize_hundreds_number(0))
        elif(number<10000):
            if(number%100==0):
                if(self.digits[0]==1 and self.digits[1]==0):
                    text="ሽሕ"
                    reading_form.append(text)
                elif(self.digits[0]!=1 and self.digits[1]==0):
                    text=self.normal_form[0]+" ሽሕ"
                    reading_form.append(text)
                elif(self.digits[0]!=1 and self.digits[1]!=0):
                    if(self.digits[1]==1):
                        text=self.normal_form[0]+" ሽሕን "+"ሚእትን"
                        reading_form.append(text)
                    else:
                        text=self.normal_form[0]+" ሽሕን "+self.normal_form[1]+" ሚእትን"
                        reading_form.append(text)
                else:
                    if(self.digits[1]==1):
                        text=" ሽሕን "+"ሚእትን"
                        reading_form.append(text)
                    else:
                        text="ሽሕን "+self.normal_form[1]+" ሚእትን"
                        reading_form.append(text)
            else:
                ##("yes")
                if(self.digits[0]==1):
                    text="ሽሕን "
                else:
                    text=self.normal_form[0]+" ሽሕን "
                reading_form.append(text)
                reading_form.append(self.normalize_hundreds_number(1))
        elif(number<100000):
            if(number%1000==0):
                if(self.digits[1]==0):
                    text=self.normal_form[0]+" ሽሕ"
                    reading_form.append(text)
                else:
                    reading_form.append(self.normalize_tenth_numbers(0))
                    reading_form.append(" ሽሕ")
            else:
                reading_form.append(self.normalize_tenth_numbers(0))
                reading_form.append(" ሽሕን ")
                reading_form.append(self.normalize_hundreds_number(2))
        elif(number<1000000):
            if(number%10000==0):
                reading_form.append(self.normalize_hundreds_number(0))
                reading_form.append(" ሽሕ ")
            else:
                reading_form.append(self.normalize_hundreds_number(0))
                if(self.digits[3]!=0 or self.digits[4]!=0 or self.digits[5]!=0):
                    reading_form.append(" ሽሕን ")
                    reading_form.append(self.normalize_hundreds_number(3))
                else:
                    reading_form.append(" ሽሕ ")
        elif(number<1000000000):
            if(number<10000000):
                point=1
                reading_form.append(self.normal_form[0])
                #("yes")
            elif(number<100000000):
                point=2
                reading_form.append(self.normalize_tenth_numbers(0))
            else:
                point=3
                reading_form.append(self.normalize_hundreds_number(0))
            if(self.digits[point]!=0 or self.digits[point+1]!=0 or self.digits[point+2]!=0):
                reading_form.append(" ሚልዮንን ")
                reading_form.append(self.normalize_hundreds_number(point))
                reading_form.append(" ሽሕን ")
                if(self.digits[point+3]!=0 or self.digits[point+4]!=0 or self.digits[point+5]!=0):
                    reading_form.append(self.normalize_hundreds_number(point+3))
            elif(self.digits[point+3]!=0 or self.digits[point+4]!=0 or self.digits[point+5]!=0):
                reading_form.append(" ሚልዮንን ")
                reading_form.append(self.normalize_hundreds_number(point+3))
            else:
                reading_form.append(" ሚልዮን ")
        elif(number<1000000000000):
            if(number<10000000000):
                point=1
                reading_form.append(self.normal_form[0])
            elif(number<100000000000):
                point=2
                reading_form.append(self.normalize_tenth_numbers(0))
            else:
                point=3
                reading_form.append(self.normalize_hundreds_number(0))
            if(self.digits[point]!=0 or self.digits[point+1]!=0 or self.digits[point+2]!=0):
                reading_form.append(" ቢልዮንን ")
                reading_form.append(self.normalize_hundreds_number(point))
                if(self.digits[point+3]!=0 or self.digits[point+4]!=0 or self.digits[point+5]!=0):
                    reading_form.append(" ሚልዮንን ")
                    reading_form.append(self.normalize_hundreds_number(point+3))
                    reading_form.append(" ሽሕን ")
                    if(self.digits[point+6]!=0 or self.digits[point+7]!=0 or self.digits[point+8]!=0):
                        reading_form.append(self.normalize_hundreds_number(point+6))
            elif(self.digits[point+3]!=0 or self.digits[point+4]!=0 or self.digits[point+5]!=0):
                reading_form.append(" ቢልዮንን ")
                reading_form.append(self.normalize_hundreds_number(point+3))
                reading_form.append(" ሽሕን ")
                if(self.digits[point+6]!=0 or self.digits[point+7]!=0 or self.digits[point+8]!=0):
                        reading_form.append(self.normalize_hundreds_number(point+6))
            elif(self.digits[point+6]!=0 or self.digits[point+7]!=0 or self.digits[point+8]!=0):
                reading_form.append(" ቢልዮንን ")
                reading_form.append(self.normalize_hundreds_number(point+6))
            else:
                reading_form.append(" ቢልዮን")
        ##(reading_form)
        return "".join(reading_form)


# In[79]:


if __name__=='__main__':
    num=Numbers()
    file=open("ekkebkab numbers.txt","r",encoding="UTF-8")
    text=file.read()
    file.close()
    text=num.normalize_per_line(text)
    file=open("normalized ekkebkab numbers.txt","w",encoding="UTF-8")
    file.write(text)
    text=num.normalize_telephone_number("+29107212222")
    #(text)
    num1=[1,10,11,202,100,101,110,11,230,367,999]
    num2=[1000,1001,1010,1100,2000,2020,2022,9999]
    num3=[10000,10001,20000,11232,23432,20454,32322]
    num4=[100000,100001,100010,201232,232234,212000]
    num5=[1000000000,1000000011,1000000010,20123200000,2322340,21210100]
    for i in num2:
        ##(i)
        k=num.normalize_numbers(0)
        ##(k)


# In[ ]:





# In[ ]:




