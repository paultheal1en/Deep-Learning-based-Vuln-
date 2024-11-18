void error_set(Error **errp, ErrorClass err_class, const char *fmt, ...)

{

    va_list ap;



    va_start(ap, fmt);

    error_setv(errp, err_class, fmt, ap);

    va_end(ap);

}
