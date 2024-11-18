void error_setg(Error **errp, const char *fmt, ...)

{

    va_list ap;



    va_start(ap, fmt);

    error_setv(errp, ERROR_CLASS_GENERIC_ERROR, fmt, ap);

    va_end(ap);

}
