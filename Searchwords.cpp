// ConsoleApplication4.cpp: 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "iostream"
#include "string"
#define strcasecmp _stricmp
#define N 50331553
FILE *fin, *fout;
char *W[1000000];
int Hash[N] = {0};
int Sign[N] = {0};
int Fren[N] = {0};
typedef struct Word {
	char c[20];
	int frequency;
}WLNode,*WLinkList;
unsigned int BKDRHash(char *str);
bool Ascll(char c);
bool isc(char c);
bool is(char c);
void charnum(void);
void wordnum(void);
void Insert(char *str,int n);
bool Compare(char *str1, char *str2);
bool Compare(char *str1, char *str2) {
	char new1[50], new2[50];
	int n1,n2,i,n;
	n1 = strlen(str1);
	n2 = strlen(str2);
	n1--;
	n2--;
	while ((int)str1[n1] > 47 && (int)str1[n1] < 58) {
		n1--;
	}
	for (i = 0; i < n1 + 1; i++)
		new1[i] = new2[i];
	new1[n1 + 1] = '\0';  //去掉str1数字结尾
	while ((int)str2[n2] > 47 && (int)str2[n2] < 58) {
		n2--;
	}
	for (i = 0; i < n2 + 1; i++)
		new2[i] = str2[i];
	new2[n2 + 1] = '\0';  //去掉str2数字结尾
	n = strcasecmp(new1, new2);
	if (n == 0)  
		return true;
	else 
		return false;
}
void Insert(char *str,int n) {
	int hashnum;
	int p,a;
	hashnum = BKDRHash(str);
	A:if (Sign[hashnum] == 1) {
		p = Hash[hashnum];
		a = Compare(W[p], str);
		if (a == 1) {
			Fren[p]++;
			return;
		}
		else {
			hashnum++;
			goto A;
		}
	}
	else {
		Hash[hashnum] = n;
		Sign[hashnum] = 1;
		Fren[hashnum] = 1;
	}
}
//char* Standard(char *str);
/*char* Standard(char *str) {
	int n;
	n = strlen(str);
		while ((int)str[n - 1] > 47 && (int)str[n - 1] < 58) {
			str[n - 1] = '\0';
			n--;
		}
		return(str);
}*/
unsigned int BKDRHash(char *str)
{
	int n,i;
	unsigned int seed = 131; // 31 131 1313 13131 131313 etc..  
	unsigned int hash = 0;
	char copy[50];
	n = strlen(str);
	n--;
	while ((int)str[n] > 47 && (int)str[n] < 58) {
		n--;
	}
	for (i = 0; i < n+1; i++)
		copy[i] = str[i];
	    copy[n + 1] = '\0';  //去掉数字结尾
		n = 0;
	while (copy[n]!='\0')
	{
		hash = hash * seed +copy[n];
		n++;
	}
	return ((hash & 0x7FFFFFFF)% 201326611);//201326611为一个大质数
}
bool Ascll(char c) {
	 if ((int)c >= 32 && (int)c <= 126)
		 return true;
	 else
		 return false;
 }
bool isc(char c) {
	if (((int)c > 47 && (int)c < 58) || ((int)c > 64 && (int)c < 91) || ((int)c > 96 && (int)c < 123))
		return true;
	else
		return false;
}
bool is(char c) {
	if (((int)c > 64 && (int)c < 91) || ((int)c > 96 && (int)c < 123))
		return true;
	else
		return false;
}
void charnum(void) {
	long int chnum = 0, linenum = 0;
	char ch;
    while ((ch = fgetc(fin)) != EOF)
	{
		if(Ascll(ch))
		   chnum++;
		if (ch == '\n')
			linenum++;
		
	}
	linenum++;
	fprintf(fout, "characters:%d \nlines: %d\n", chnum, linenum);
}
void wordnum(void) {
	int hashnum,p;
	FILE *fin1, *fin2, *fin3;
	char ch,ch1,ch2,ch3; 
     int a,k,i=0,wordmax=0,n;
	long int wordsnum = 0;
	unsigned int h,hashmax = 0;
	a = fopen_s(&fin1, "Aesop’s Fables.txt", "r");// printf("%d", a);
	a = fopen_s(&fin2, "Aesop’s Fables.txt", "r");// printf("%d", a);
	a = fopen_s(&fin3, "Aesop’s Fables.txt", "r"); //printf("%d", a);
	ch2 = fgetc(fin2);
	ch1 = fgetc(fin1);
	ch1 = fgetc(fin1);
	ch = fgetc(fin); 
	ch = fgetc(fin); 
	ch = fgetc(fin);
	ch3 = fgetc(fin3); //putchar(ch3);
	ch2 = fgetc(fin2); //putchar(ch2);
	ch1 = fgetc(fin1); //putchar(ch1);
	ch  = fgetc(fin) ; //putchar(ch) ;
	while (ch != EOF) {
		if (is(ch1) && is(ch2) && is(ch3) && is(ch)) {
			wordsnum++;
			//n = 3;
			k = 0;
			W[i] = (char *)malloc(50*sizeof(char));
			W[i][k] = ch3; 
			k++;
			W[i][k] = ch2; 
			k++;
			W[i][k] = ch1; 
			k++;
				while (isc(ch)) {
					W[i][k] = ch; k++;
					ch3 = fgetc(fin3); 
				    ch2 = fgetc(fin2); 
				    ch1 = fgetc(fin1); 
				    ch  = fgetc(fin) ; 
					//n++;
			}
				W[i][k] = '\0';
				i++;
				Insert(W[i],i);
				//if (n > wordmax)  wordmax = n;
		}
		ch3 = fgetc(fin3);
		ch2 = fgetc(fin2);
		ch1 = fgetc(fin1);
		ch = fgetc(fin);
	}
	fprintf(fout, "words:%d\n", wordsnum);
	//fprintf(fout, "wordmax:%d", wordmax);
	/*for (n = 0; n < i; n++) {
		h = BKDRHash(W[n]);
		fprintf(fout, "%u，", h);
		fprintf(fout, "%s\n", W[n]);
		if (h> hashmax)
			hashmax = h;
	}*/
	//fprintf(fout, "%u\n", hashmax);
	hashnum = BKDRHash(W[0]);
	p = Hash[hashnum];
	fprintf(fout, "%d\n", Fren[p]);
	putchar('\n');
	fclose(fin1);
	fclose(fin2);
	fclose(fin3);
}
int main()
{
	
	errno_t err;
	err = fopen_s(&fin, "Aesop’s Fables.txt ","r");
	if (err == 0)
	{
		printf("The file'xxx.in'was opened\n");
	}
	else
	{
		printf("The file'xxx.in'was not opened\n");
	}
	err = fopen_s(&fout, "result.txt", "w");
	if (err == 0)
	{
		printf("The file'xxx.out'was opened\n");
	}
	else
	{
		printf("The file'xxx.out'was not opened\n");
	}
	charnum();
	rewind(fin);
	wordnum();
	fclose(fin);
	fclose(fout);
	system("pause");
    return 0;
}

