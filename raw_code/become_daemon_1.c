static void become_daemon(const char *pidfile)

{

#ifndef _WIN32

    pid_t pid, sid;



    pid = fork();

    if (pid < 0) {

        exit(EXIT_FAILURE);

    }

    if (pid > 0) {

        exit(EXIT_SUCCESS);

    }



    if (pidfile) {

        if (!ga_open_pidfile(pidfile)) {

            g_critical("failed to create pidfile");

            exit(EXIT_FAILURE);

        }

    }



    umask(0);

    sid = setsid();

    if (sid < 0) {

        goto fail;

    }

    if ((chdir("/")) < 0) {

        goto fail;

    }



    close(STDIN_FILENO);

    close(STDOUT_FILENO);

    close(STDERR_FILENO);

    return;



fail:

    unlink(pidfile);

    g_critical("failed to daemonize");

    exit(EXIT_FAILURE);

#endif

}
