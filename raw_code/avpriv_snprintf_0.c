int avpriv_snprintf(char *restrict s, size_t n, const char *restrict fmt, ...)

{

    va_list ap;

    int ret;



    va_start(ap, fmt);

    ret = avpriv_vsnprintf(s, n, fmt, ap);

    va_end(ap);



    return ret;

}
