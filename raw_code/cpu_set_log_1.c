void cpu_set_log(int log_flags)

{

    loglevel = log_flags;

    if (loglevel && !logfile) {

        logfile = fopen(logfilename, log_append ? "a" : "w");

        if (!logfile) {

            perror(logfilename);

            _exit(1);

        }

#if !defined(CONFIG_SOFTMMU)

        /* must avoid mmap() usage of glibc by setting a buffer "by hand" */

        {

            static char logfile_buf[4096];

            setvbuf(logfile, logfile_buf, _IOLBF, sizeof(logfile_buf));

        }

#elif !defined(_WIN32)

        /* Win32 doesn't support line-buffering and requires size >= 2 */

        setvbuf(logfile, NULL, _IOLBF, 0);

#endif

        log_append = 1;

    }

    if (!loglevel && logfile) {

        fclose(logfile);

        logfile = NULL;

    }

}
