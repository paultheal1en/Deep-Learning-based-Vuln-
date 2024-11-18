static void error(const char *fmt, ...)

{

    va_list ap;

    va_start(ap, fmt);

    fprintf(stderr, "qemu-img: ");

    vfprintf(stderr, fmt, ap);

    fprintf(stderr, "\n");

    va_end(ap);

}
