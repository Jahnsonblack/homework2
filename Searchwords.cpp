// ConsoleApplication4.cpp: 定义控制台应用程序的入口点。
//
/*总思路：hashnum存储字符串的哈希值，用Hash的下标来表示，Hash此下标中存
着字符指针的下标（即W[n]中的n），同时Fren[hashnum]存储对应Hash[hashnum]的频率
所以最终输出时对Fren数组进行查找出最大的，然后找出对应Hash表中的字符指针下表*/
#include "stdafx.h"
#include "iostream"
#include "string"
#define strcasecmp _stricmp
#define N 50331553
FILE *fin, *fout;
char *W[20000000];
char *P[20000000];
int Hash[N] = {0};
int Fren[N] = {0};
int HashP[N] = {0};
int FrenP[N] = {0};
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
int Insert(char *str,int n);
int InsertP(char *str, int n);
bool Compare(char *str1, char *str2);
bool CompareP(char *str1, char *str2);
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
		new1[i] = str1[i];
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
int  Insert(char *str,int n) {
    int hashnum;
	int p,a;
	hashnum = BKDRHash(str); //printf("%d",hashnum);
	while(Fren[hashnum]>0) {
		//printf("%s\n", str);
		p = Hash[hashnum];
		a = Compare(W[p],str); //printf("%d", a);
		if (a == 1) {
			if (strcmp(W[p], str)> 0) {
				Hash[hashnum] = n;
				Fren[hashnum]++;
				return 0;
			}
			Fren[hashnum]++;
			return 1;
		}
		else 
			hashnum=(hashnum+1)%N;
	}
	
		Hash[hashnum] = n;
		Fren[hashnum] = 1;
		return 0;
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
		while (copy[n] != '\0')
		{
			if ((int)copy[n] > 96 && (int)copy[n] < 123)
				copy[n] = copy[n] - 32;
		hash = hash * seed +copy[n];
		n++;
	}
	return ((hash & 0x7FFFFFFF)% N);//N为一个大质数
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
	//int hashnum;
	FILE *fin1, *fin2, *fin3;
	char ch,ch1,ch2,ch3; 
     int a,k,i=0,wordmax=0,n,sign,p=0,q,l;//p,q与i,k相对应是词组的下标
	long int wordsnum = 0;
	unsigned int hashmax = 0;
	a = fopen_s(&fin1, "Aesop’s Fables.txt", "r"); //printf("%d", a);
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
	P[0]= (char *)malloc(100 * sizeof(char));
	P[0][0] = '\0';
	while (ch != EOF) {
		if (is(ch1) && is(ch2) && is(ch3) && is(ch)) {
			wordsnum++;
			//n = 3;
			k = 0; q = 0; l = strlen(P[p]);
			W[i] = (char *)malloc(50*sizeof(char));
			P[p+1]= (char *)malloc(100 * sizeof(char));
			W[i][k] = ch3;
			P[p][l] = ch3;
			P[p + 1][q] = ch3;
			k++; q++; l++;
			W[i][k] = ch2; 
			P[p][l] = ch2;
			P[p + 1][q] = ch2;
			k++; q++; l++;
			W[i][k] = ch1; 
			P[p][l] = ch1;
			P[p + 1][q] = ch1;
			k++; q++; l++;
				while (isc(ch)) {
					W[i][k] = ch;
					P[p][l] = ch;
					P[p + 1][q] = ch;
					k++; q++; l++;
					ch3 = fgetc(fin3); 
				    ch2 = fgetc(fin2); 
				    ch1 = fgetc(fin1); 
				    ch  = fgetc(fin) ; 
					//n++;
			}
				W[i][k] = '\0';
				P[p][l] = '\0';
				P[p + 1][q] = ' ';
				q++;
				P[p + 1][q] = '\0';
				fprintf(fout, "%s \n", P[p]);
				sign=Insert(W[i],i);
				
				if (sign == 1) {
					free(W[i]);
				}
				else i++;
				//if (n > wordmax)  wordmax = n;
				p++;
		}
		ch3 = fgetc(fin3);
		ch2 = fgetc(fin2);
		ch1 = fgetc(fin1);
		ch = fgetc(fin);
	}
	int x, y, r1,r2;
	for(x=0;x<10;x++)
		for (y = x+1; y < N; y++)
			if (Fren[x] < Fren[y]) {
				r1 = Fren[x];
				Fren[x] = Fren[y];
				Fren[y] = r1;
				r2 = Hash[x];
				Hash[x] = Hash[y];
				Hash[y] = r2;
			}
	fprintf(fout, "words:%d\n", wordsnum);
	putchar('\n');
	for (i = 0; i < 10; i++)
		fprintf(fout, "%s:%d\n",W[Hash[i]],Fren[i]);
	fclose(fin1);
	fclose(fin2);
	fclose(fin3);
}
int main(){
	errno_t err;
	err = fopen_s(&fin, "Aesop’s Fables.txt","r");
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

