void os_daemonize(void)

{

    if (daemonize) {

        pid_t pid;

        int fds[2];



        if (pipe(fds) == -1) {

            exit(1);

        }



        pid = fork();

        if (pid > 0) {

            uint8_t status;

            ssize_t len;



            close(fds[1]);



        again:

            len = read(fds[0], &status, 1);

            if (len == -1 && (errno == EINTR)) {

                goto again;

            }

            if (len != 1) {

                exit(1);

            }

            else if (status == 1) {

                fprintf(stderr, "Could not acquire pidfile\n");

                exit(1);

            } else {

                exit(0);

            }

            } else if (pid < 0) {

                exit(1);

            }



        close(fds[0]);

        daemon_pipe = fds[1];

        qemu_set_cloexec(daemon_pipe);



        setsid();



        pid = fork();

        if (pid > 0) {

            exit(0);

        } else if (pid < 0) {

            exit(1);

        }

        umask(027);



        signal(SIGTSTP, SIG_IGN);

        signal(SIGTTOU, SIG_IGN);

        signal(SIGTTIN, SIG_IGN);

    }

}
