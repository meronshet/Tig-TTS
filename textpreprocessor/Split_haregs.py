#!/usr/bin/env python
# coding: utf-8

# In[26]:


import re
class Splitter:
    #class Constructor recieve text read from file as input initialize sentences list objects 
    def __init__(self,text):
        self.text=text
        self.Hareg_list=[]
        self.Sample_list=[]
    def colon_problem(self):
        p=re.compile(r"[0-9]+[፡|።][0-9]+")
        l=p.findall(self.text)
        for i in l:
            str1=i.replace("።",":")
            str2=str1.replace("፡",":")
            p=re.compile(i)
            self.text=p.sub(str2,self.text)
        p=re.compile(r"\(..+\)")
        l=p.findall(self.text)
        for i in l:
            str1=i.replace("።"," ")
            str2=str1.replace("፡"," ")
            str3=str2.replace(r"((",r"(")
            str4=str3.replace(r"))",r")")
            str5=i.replace(r"(","\\(")
            str6=str5.replace(r")","\\)")
            p=re.compile(str6)
            self.text=p.sub(str4,self.text)
    #use ፡ or ፣ to extract meaningful sentences from long sentences
    def clause_tokenizer(self,sentence):
        serez1='፣'
        serez2='፡'
        serez1Pos=sentence.find(serez1)
        serez2Pos=sentence.find(serez2)
        clauselist=[]
        bulletPos=sentence.find(r"•")
        finaclauselist=[]
        if(bulletPos!=-1):
            newbulletedlist=[]
            bulletedlist=sentence.split(r"•")
            for i in bulletedlist:
                temp=r"• "+ i
                newbulletedlist.append(temp)
            str1="\n".join(newbulletedlist)
            clauselist.append(str1)
        elif(serez1Pos!=-1 and serez2Pos==-1):
            clauselist=sentence.split(serez1)
            i=0
            while(i<len(clauselist)-1):
                str=clauselist[i]
                clauselist[i]=str+" ፣"
                i=i+1
        elif(serez2Pos!=-1 and serez1Pos==-1):
            clauselist=sentence.split(serez2)
            i=0
            while(i<len(clauselist)-1):
                str=clauselist[i]
                clauselist[i]=str+" ፡"
                i=i+1
        elif(serez1Pos!=-1 and serez2Pos!=-1):
            clauselist=sentence.split(serez1)
            i=0
            newclauselist1=[]
            while(i<=len(clauselist)-1):
                str=clauselist[i]
                clauselist[i]=str+" ፣"
                i=i+1
                clauselist1=str.split(serez2)
                j=0
                if(len(clauselist1)>1):
                    while(j<=len(clauselist1)-1):
                        str1=clauselist1[j]
                        clauselist1[j]=str1+" ፡"
                        j=j+1
                for k in clauselist1:
                    newclauselist1.append(k)
            clauselist=newclauselist1
        else:
            clauselist.append(sentence)
        newclauselist=clauselist
        #print(True)
        #print(newclauselist
        #print(clauselist)
        str=""
        rebuild=[]
        finalHaregs=[]
        before=""
        flag=False
        j=0
        if(len(clauselist)>1):
            for i in clauselist:
                #print(i)
                #print(j)
                if(len(i)< 25 ):
                    #print(1)
                    #print(before)
                    rebuild.append(i)
                    flag=True
                else:
                    if(before=="" and j!=(len(clauselist)-1)):
                        #print(2)
                        before=i
                        rebuild.append(i)
                    else:
                        if(flag==True):
                            #print(3)
                            rebuild.append(i)
                            str="".join(rebuild)
                            finalHaregs.append(str)
                            rebuild=[]
                            before=""
                            flag=False
                        else:
                            #print(4)
                            str="".join(rebuild)
                            finalHaregs.append(str)
                            rebuild=[]
                            before=i
                            rebuild.append(i)
                j=j+1
            if(rebuild!=[]):
                str="".join(rebuild)
                finalHaregs.append(str)
        else:
            finalHaregs.append(clauselist[0])
        return finalHaregs
    #split sentences based on the stop word ። into seperate sentences then use ፡ or ፣ to extract meaningful sentences from long
    #sentences
    def space_remover(self):
        max=len(self.Hareg_list)
        i=0
        while(i<max-1):
            if (len(self.Hareg_list[i])< 5):
                self.Hareg_list.pop(i)
                max-=1
            i+=1
    def paragraph_tokenizer(self):
        Haregs=[]
        self.text.replace(r"?",r" ?")
        self.text.replace(r"!",r" !")
        self.text.replace(r"፤",r" ፤")
        lines=self.text.split('\n')     
        ts=" ".join(lines)
        p=re.compile(r'፣፣')
        ts=p.sub("።",ts)
        p=re.compile(r'፡፡')
        ts=p.sub("።",ts)
        sentences=ts.split('።')
        #print("ምሉእ ሓሳባት ተረኺቦም ኣለዉ")
        rebuild=[]
        muluhasabat=[]
        for i in sentences:
            if(len(i)< 30):
                rebuild.append(i)
            else:
                rebuild.append(i)
                str="፡ ".join(rebuild)
                muluhasabat.append(str)
                rebuild=[]
                before=""
        #print("ሓጸርቲ ሓሳባት ብዘይግቡእ ዝተጻሕፉ ተለልዮም")
        for i in muluhasabat:
            i=i.__add__(r" ።")
            splitted=self.clause_tokenizer(i)
            result="\n".join(splitted)
            #print(result)
            Haregs.append(result)
        #print("ቀንዲ ሓረግ ምስ ንኡስን ጽጉዕ ሓረጉ ካብ ካልእ ቀንዲ ሓረግ ተፈልዩ")
        filteredHareg=Haregs
        output="\n".join(filteredHareg)
        p=re.compile(r"\s*\n+\s*")
        output=p.sub(r"\n",output)
        p=re.compile(r"\s\s+")
        clean=p.sub(r" ",output)
        self.Hareg_list=clean.split("\n")
        #print(len(self.Hareg_list))
        self.space_remover()
        #stores sentences in filteredharegList variable of the class which have length less than the specified max_len integer
    def average_sentences(self,max_len):
        filteredHareg=[]
        Final_list=self.Hareg_list
        for i in Final_list:
            if(len(i)<max_len):
                filteredHareg.append(i)
            self.Hareg_list=filteredHareg
    def get_Haregs(self):
        return "\n".join(self.Hareg_list)
    def get_SampleHaregs(self):
        return "\n".join(self.Sample_list)
    def average_length(self):
        total=0
        for i in self.Hareg_list:
            total+=len(i)
        average=(total/len(self.Hareg_list))//1
        return average
    def freq_dist_of_hareg_lengths(self):
     #class_1 =1-50,class_2 =51-100,class_3 =101-150,class_1 =151-200,class_1 = >200
        frequency={"class_1":0,"class_2":0,"class_3":0,"class_4":0,"class_5":0}
        for i in self.Hareg_list:
            length=len(i)
            if(length<=50):
                key="class_1"
            elif(length>50 and length<=100):
                key="class_2"
            elif(length>100 and length<=150):
                key="class_3"
            elif(length>150 and length<=200):
                key="class_4"
            elif(length>200):
                key="class_5"
            for j in frequency.keys():
                    if(j==key):
                        count=frequency[key]
                        count=count+1
                        frequency[key]=count
                        break
        print(frequency.items())
    def Hareg_Sampler(self,number_of_haregs):#number of sentences to be sampled
        Total_length=len(self.Hareg_list)
        if(number_of_haregs<Total_length)*(95/100):
            sentence_per_one_percent=Total_length//100
            if(sentence_per_one_percent>1000):
                appropraite_percent=0.01
            elif(sentence_per_one_percent>100):
                appropraite_percent=0.1
            elif(sentence_per_one_percent>10):
                appropraite_percent=1
            elif(sentence_per_one_percent>=1):
                appropraite_percent=10
            else:
                appropraite_percent=50
            partition=(int)(Total_length*(appropraite_percent/100))+1//1
            print(partition)
            ratio=number_of_haregs/Total_length
            fold=(int)((partition*ratio)//1)
            print(fold)
            fold_pointer=0
            i=0
            while(i<Total_length):
                if(i<fold_pointer+fold):
                    self.Sample_list.append(self.Hareg_list[i])
                    i+=1
                else:
                    fold_pointer+=partition
                    i=fold_pointer
        else:
            #print("yes")
            self.Sample_list=self.Hareg_list
    def sentence_anotator(self):
        Final_list=[]
        if(self.Hareg_list!=[]):
            Final_list=self.Hareg_list
            anotated_list=[]
            for i in range(1,len(Final_list)):
                string= str(i)+" "+Final_list[i-1]
                anotated_list.append(string)
            self.Hareg_list=anotated_list
            if(self.Sample_list!=[]):
                Final_list=self.Sample_list
                anotated_list=[]
                for i in range(1,len(Final_list)):
                    string= str(i)+" "+Final_list[i-1]
                    anotated_list.append(string)
                self.Sample_list=anotated_list
        else:
            print("error: hareg list is empty")


# In[27]:


if __name__ == '__main__':
    print("enter the name of files to edit in comma seperated list")
    oldfilelist=input("enter old filelist:")
    files=oldfilelist.split(",")
    for i in files:
        file=open(i+".txt","r",encoding="UTF-8")
        newfile=open("split domains/new-"+i+".txt","w",encoding="UTF-8")
        text=file.read()
        file.close()
        Splitter_obj=Splitter(text)
        Splitter_obj.colon_problem()
        Splitter_obj.colon_problem()
        Splitter_obj.paragraph_tokenizer()
        length=Splitter_obj.average_length()
        Splitter_obj.freq_dist_of_hareg_lengths()
        print(length)
        ts=Splitter_obj.get_Haregs()
        newfile.write(ts)
        newfile.close()


# In[ ]:





# In[ ]:





# In[ ]:





# In[6]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




