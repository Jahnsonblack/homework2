#include<stdio.h>
#include <stdlib.h> 
#include <time.h>  
#include<string.h>
char S[1024];
   // int numOfQuestion;		//题目的数量
	int numOfOperand;		//操作数的数量
	double rangeOfAnswer;	//答案中的数值范围
	double rangeOfQuestion;  //题目中的数值范围 
	bool addition;
	bool subtraction;
	bool multiplication;
	bool division;
	bool power;
	bool brackets;
	bool properFraction;	//是否支持真分数
	bool decimalFraction;	//是否支持小数
							//都不支持说明支持整数
int getrandom(int n);
void generate1(void);
void generate2(void);
void generate3(void);
void generate(void); 
int Priority(char a);
int getrandom(int n){
   srand((unsigned)time(NULL)+rand()); 
   return(rand()%n+1);
}
int Priority(char a)
{
	switch (a) {
	case '+':return 1; break;
	case '-':return 1; break;
	case '*':return 2; break;
	case '/':return 2; break;
	case '^':return 3; break;
	case '#':return 0; break;
	case '\n':return 0; break;
	case '(':return 0; break;
	}
}
void generate(){
	if(properFraction==0&&decimalFraction==0){
		generate1();
		return;
	}
	if(properFraction==1&&decimalFraction==0){
		generate2();
		return;
	}
	if(properFraction==0&&decimalFraction==1){
		generate3();
		return;
	}
}
void generate1(){
	int Operand[512];
	char Operator[512];
	int numOfOperator;     //操作符的数量
	numOfOperator=numOfOperand-1;
	int i=0,j,length=0;
	char s[1024];
	int random;
	while(i<numOfOperator){
		
		switch(getrandom(5)){
		case 1 :  if(addition==1)  {
		            Operator[i]='+';
		            i++;
		            break;
		        }
		case 2 :  if(subtraction==1)  {
		            Operator[i]='-';
		            i++; 
					break;  
		        } 
		case 3 :  if(multiplication==1)  {
		            Operator[i]='*';
		            i++;  
		            break;
		        } 
		case 4 :  if(division==1)  {
		            Operator[i]='/';
		            i++; 
		            break;
		        }  
		case 5 :  if(power==1)  {
		            Operator[i]='^';
		            i++;
		            break;
		        }    
			}
		
	}
	Operator[i]='\0';
	i=0;
	while(i<numOfOperand){
		Operand[i]=getrandom((int)rangeOfQuestion);
		i++;
	}
	i=0;
	while(i<numOfOperand-1){
		char str[512];
		itoa(Operand[i],str,10);
	    strcat(S,str);
	    if(Operator[i]=='^'){
	    	random=getrandom(2);
	    	random++;
	    	Operand[i+1]=random;
	    }
	    if(brackets==1&&i>=1&&(Priority(Operator[i])>Priority(Operator[i-1]))){
	    	s[0]='(';
	    	s[1]='\0';
	    	strcat(s,S);
	    	length=strlen(s);
	    	s[length]=')';
	    	s[length+1]='\0';
            strcpy(S,s);
            for(j=0;j<1024;j++)
            s[j]='\0';
	    }
	    length=strlen(S);
	    S[length]=Operator[i];
		i++;
	}
	    char str[512];
	    itoa(Operand[i],str,10);
	    strcat(S,str);
	  puts(S); 
	   
}
  void generate2(){
	
	int numerator[512];    //分子
	int denominator[512];   //分母 
	char Operator[512];
	int numOfOperator;     //操作符的数量
	numOfOperator=numOfOperand-1;
	int i=0,j,length=0,n;
	char s[1024];
	int random;
	while(i<numOfOperator){
		
		switch(getrandom(5)){
		case 1 :  if(addition==1)  {
		            Operator[i]='+';
		            i++;
		            break;
		        }
		case 2 :  if(subtraction==1)  {
		            Operator[i]='-';
		            i++; 
					break;  
		        } 
		case 3 :  if(multiplication==1)  {
		            Operator[i]='*';
		            i++;  
		            break;
		        } 
		case 4 :  if(division==1)  {
		            Operator[i]='/';
		            i++; 
		            break;
		        }  
		case 5 :  if(power==1)  {
		            Operator[i]='^';
		            i++;
		            break;
		        }    
			}
		
	}
	Operator[i]='\0';
	i=0;
	while(i<numOfOperand){
		numerator[i]=getrandom((int)rangeOfQuestion);
		denominator[i]=getrandom(50);
		i++;
	}
	i=0;
	while(i<numOfOperand-1){
		char str[512];
		if(numerator[i]>=denominator[i]){                        //当分子大于分母时，化为真分数 
			if(numerator[i]%denominator[i]==0){
				n=numerator[i]/denominator[i];
				itoa(n,str,10);
				strcat(S,str);
			}
			else{
				n=numerator[i]/denominator[i];
				itoa(n,str,10);
				length=strlen(str);
				str[length]=39;
				str[length+1]='\0';
				strcat(S,str);
				n=numerator[i]-n*denominator[i];
				itoa(n,str,10);
				length=strlen(str);
				str[length]='/';
				str[length+1]='\0';
				strcat(S,str);
				itoa(denominator[i],str,10);
				strcat(S,str);
			}
		}
		else{                                                    //当分子小于分母时 ，直接输出 
			itoa(numerator[i],str,10);
			length=strlen(str);
			str[length]='/';
			str[length+1]='\0';
			strcat(S,str);
			itoa(denominator[i],str,10);
			strcat(S,str);
	}
	    if(Operator[i]=='^'){
	    	random=getrandom(2);
	    	random++;
	    	numerator[i+1]=random;
	    	denominator[i+1]=1; 
	    }
	    if(brackets==1&&i>=1&&(Priority(Operator[i])>Priority(Operator[i-1]))){
	    	s[0]='(';
	    	s[1]='\0';
	    	strcat(s,S);
	    	length=strlen(s);
	    	s[length]=')';
	    	s[length+1]='\0';
            strcpy(S,s);
            for(j=0;j<1024;j++)
            s[j]='\0';
	    }
	    length=strlen(S);
	    S[length]=Operator[i];
		i++;
	}
	    char str[512];
		if(numerator[i]>=denominator[i]){                        //当分子大于分母时，化为真分数 
			if(numerator[i]%denominator[i]==0){
				n=numerator[i]/denominator[i];
				itoa(n,str,10);
				strcat(S,str);
			}
			else{
				n=numerator[i]/denominator[i];
				itoa(n,str,10);
				length=strlen(str);
				str[length]=39;
				str[length+1]='\0';
				strcat(S,str);
				n=numerator[i]-n*denominator[i];
				itoa(n,str,10);
				length=strlen(str);
				str[length]='/';
				str[length+1]='\0';
				strcat(S,str);
				itoa(denominator[i],str,10);
				strcat(S,str);
			}
		}
		else{                                                    //当分子小于分母时 ，直接输出 
			itoa(numerator[i],str,10);
			length=strlen(str);
			str[length]='/';
			str[length+1]='\0';
			strcat(S,str);
			itoa(denominator[i],str,10);
			strcat(S,str);
		}
	   puts(S);
}
void generate3(){
	int Integralpart[512];   //整数部分 
	int  Decimalpart[512];   //小数部分 
	char Operator[512];
	int numOfOperator;     //操作符的数量
	numOfOperator=numOfOperand-1;
	int i=0,j,length=0;
	char s[1024];
	int random;
	while(i<numOfOperator){
		
		switch(getrandom(5)){
		case 1 :  if(addition==1)  {
		            Operator[i]='+';
		            i++;
		            break;
		        }
		case 2 :  if(subtraction==1)  {
		            Operator[i]='-';
		            i++; 
					break;  
		        } 
		case 3 :  if(multiplication==1)  {
		            Operator[i]='*';
		            i++;  
		            break;
		        } 
		case 4 :  if(division==1)  {
		            Operator[i]='/';
		            i++; 
		            break;
		        }  
		case 5 :  if(power==1)  {
		            Operator[i]='^';
		            i++;
		            break;
		        }    
			}
		
	}
	Operator[i]='\0';
	i=0;
	while(i<numOfOperand){
		Integralpart[i]=getrandom((int)rangeOfQuestion-1);
		Decimalpart[i]=getrandom(99);
		i++;
	}
	i=0;
	while(i<numOfOperand-1){
		char str[512];
		if(Decimalpart[i]==0){
			itoa(Integralpart[i],str,10);
			strcat(S,str);
		}
		else{
		itoa(Integralpart[i],str,10);
		length=strlen(str);
		str[length]='.';
		str[length+1]='\0';
	    strcat(S,str);
	    itoa(Decimalpart[i],str,10);
	    strcat(S,str);
	}
	    if(Operator[i]=='^'){
	    	random=getrandom(2);
	    	random++;
	    	Integralpart[i+1]=random;
	    	Decimalpart[i+1]=0;
	    }
	    if(brackets==1&&i>=1&&(Priority(Operator[i])>Priority(Operator[i-1]))){
	    	s[0]='(';
	    	s[1]='\0';
	    	strcat(s,S);
	    	length=strlen(s);
	    	s[length]=')';
	    	s[length+1]='\0';
            strcpy(S,s);
            for(j=0;j<1024;j++)
            s[j]='\0';
	    }
	    length=strlen(S);
	    S[length]=Operator[i];
		i++;
	}
	   char str[512];
		if(Decimalpart[i]==0){
			itoa(Integralpart[i],str,10);
			strcat(S,str);
		}
		else{
		itoa(Integralpart[i],str,10);
		length=strlen(str);
		str[length]='.';
		str[length+1]='\0';
	    strcat(S,str);
	    itoa(Decimalpart[i],str,10);
	    strcat(S,str);
	}
	    puts(S);
}
main(){
    addition=1;
    subtraction=1;
    multiplication=1;
    division=1;
    power=1;
	numOfOperand=4;
	brackets=0;
	rangeOfQuestion=100;
	properFraction=0;
	decimalFraction=1;
    generate();
}
