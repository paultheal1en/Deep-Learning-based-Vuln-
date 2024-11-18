static char *check_nan_suffix(char *s)

{

    char *start = s;



    if (*s++ != '(')

        return start;



    while ((*s >= 'a' && *s <= 'z') || (*s >= 'A' && *s <= 'Z') ||

           (*s >= '0' && *s <= '9') ||  *s == '_')

        s++;



    return *s == ')' ? s + 1 : start;

}
