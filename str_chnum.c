#include<stdio.h>
int str_chnum(const char a[],int c){
    int i;
    int b=0;
    for(i=0;a[i]!='\0';i++){
        if(a[i]=='c'){
        b++;
        }
    }
    return b;
}
int main(){
    char s[128];
    int c=0;
    printf("調べる文字列：");
    scanf("%s",s);
    printf("cは%d個あります。\n",str_chnum(s,c));
}