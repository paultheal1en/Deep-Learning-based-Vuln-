void qemu_set_log_filename(const char *filename)

{

    char *pidstr;

    g_free(logfilename);



    pidstr = strstr(filename, "%");

    if (pidstr) {

        /* We only accept one %d, no other format strings */

        if (pidstr[1] != 'd' || strchr(pidstr + 2, '%')) {

            error_report("Bad logfile format: %s", filename);

            logfilename = NULL;

        } else {

            logfilename = g_strdup_printf(filename, getpid());

        }

    } else {

        logfilename = g_strdup(filename);

    }

    qemu_log_close();

    qemu_set_log(qemu_loglevel);

}
