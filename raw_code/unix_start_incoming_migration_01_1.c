int unix_start_incoming_migration(const char *path)

{

    struct sockaddr_un un;

    int sock;



    dprintf("Attempting to start an incoming migration\n");



    sock = socket(PF_UNIX, SOCK_STREAM, 0);

    if (sock < 0) {

        fprintf(stderr, "Could not open unix socket: %s\n", strerror(errno));

        return -EINVAL;

    }



    memset(&un, 0, sizeof(un));

    un.sun_family = AF_UNIX;

    snprintf(un.sun_path, sizeof(un.sun_path), "%s", path);



    unlink(un.sun_path);

    if (bind(sock, (struct sockaddr*) &un, sizeof(un)) < 0) {

        fprintf(stderr, "bind(unix:%s): %s\n", un.sun_path, strerror(errno));

        goto err;

    }

    if (listen(sock, 1) < 0) {

        fprintf(stderr, "listen(unix:%s): %s\n", un.sun_path, strerror(errno));

        goto err;

    }



    qemu_set_fd_handler2(sock, NULL, unix_accept_incoming_migration, NULL,

			 (void *)(unsigned long)sock);



    return 0;



err:

    close(sock);



    return -EINVAL;

}
