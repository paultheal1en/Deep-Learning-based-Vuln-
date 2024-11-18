static void guest_suspend(const char *pmutils_bin, const char *sysfile_str,

                          Error **err)

{

    pid_t pid;

    char *pmutils_path;



    pmutils_path = g_find_program_in_path(pmutils_bin);



    pid = fork();

    if (pid == 0) {

        /* child */

        int fd;



        setsid();

        reopen_fd_to_null(0);

        reopen_fd_to_null(1);

        reopen_fd_to_null(2);



        if (pmutils_path) {

            execle(pmutils_path, pmutils_bin, NULL, environ);

        }



        /*

         * If we get here either pm-utils is not installed or execle() has

         * failed. Let's try the manual method if the caller wants it.

         */



        if (!sysfile_str) {

            _exit(EXIT_FAILURE);

        }



        fd = open(LINUX_SYS_STATE_FILE, O_WRONLY);

        if (fd < 0) {

            _exit(EXIT_FAILURE);

        }



        if (write(fd, sysfile_str, strlen(sysfile_str)) < 0) {

            _exit(EXIT_FAILURE);

        }



        _exit(EXIT_SUCCESS);

    }



    g_free(pmutils_path);



    if (pid < 0) {

        error_set(err, QERR_UNDEFINED_ERROR);

        return;

    }

}
