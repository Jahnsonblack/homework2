// ConsoleApplication4.cpp: 定义控制台应用程序的入口点。
//

#include "stdafx.h"
FILE *fin, *fout;
 int wordsnum = 0;
void charnum(void) {
	int chnum = 0, linenum = 0;
	char ch;
	while ((ch = fgetc(fin)) != EOF)
	{
		//printf("%c", ch);
		chnum++;
		if (ch == '\n')
			linenum++;
	}
	linenum++;
	fprintf(fout, "字符总数：%d \n换行符总数:%d\n", chnum, linenum);
}
void wordnum(void) {
	char ch;
	while ((ch = fgetc(fin)) != EOF)
	{
	}
}
int main()
{
	errno_t err;
	err = fopen_s(&fin, "Aesop’s Fables.txt", "r");
	if (err == 0)
	{
		printf("The file'xxx.in'was opened\n");
	}
	else
	{
		printf("The file'xxx.in'was not opened\n");
	}
	err = fopen_s(&fout, "xxx.out", "w");
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
	//wordnum();
	
	fclose(fin);
	fclose(fout);
	getchar();
    return 0;
}

