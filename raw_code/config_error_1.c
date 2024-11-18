static void config_error(Monitor *mon, const char *fmt, ...)

{

    va_list ap;



    va_start(ap, fmt);

    if (mon) {

        monitor_vprintf(mon, fmt, ap);

    } else {

        fprintf(stderr, "qemu: ");

        vfprintf(stderr, fmt, ap);

        exit(1);

    }

    va_end(ap);

}
