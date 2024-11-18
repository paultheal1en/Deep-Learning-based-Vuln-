static void GCC_FMT_ATTR(2, 3) qtest_sendf(CharBackend *chr,

                                           const char *fmt, ...)

{

    va_list ap;

    gchar *buffer;



    va_start(ap, fmt);

    buffer = g_strdup_vprintf(fmt, ap);

    qtest_send(chr, buffer);


    va_end(ap);

}