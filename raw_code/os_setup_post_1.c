void os_setup_post(void)

{

    int fd = 0;



    if (daemonize) {

        uint8_t status = 0;

        ssize_t len;



        do {        

            len = write(daemon_pipe, &status, 1);

        } while (len < 0 && errno == EINTR);

        if (len != 1) {

            exit(1);

        }

        if (chdir("/")) {

            perror("not able to chdir to /");

            exit(1);

        }

        TFR(fd = qemu_open("/dev/null", O_RDWR));

        if (fd == -1) {

            exit(1);

        }

    }



    change_root();

    change_process_uid();



    if (daemonize) {

        dup2(fd, 0);

        dup2(fd, 1);

        dup2(fd, 2);



        close(fd);

    }

}
