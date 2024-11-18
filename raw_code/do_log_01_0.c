static void do_log(int loglevel, const char *format, ...)

{

    va_list ap;



    va_start(ap, format);

    if (is_daemon) {

        vsyslog(LOG_CRIT, format, ap);

    } else {

        vfprintf(stderr, format, ap);

    }

    va_end(ap);

}
