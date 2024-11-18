static void monitor_puts(Monitor *mon, const char *str)

{

    char c;



    for(;;) {

        c = *str++;

        if (c == '\0')

            break;

        if (c == '\n') {

            qstring_append_chr(mon->outbuf, '\r');

        }

        qstring_append_chr(mon->outbuf, c);

        if (c == '\n') {

            monitor_flush(mon);

        }

    }

}
