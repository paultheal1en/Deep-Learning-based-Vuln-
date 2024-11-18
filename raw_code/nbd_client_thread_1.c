static void *nbd_client_thread(void *arg)

{

    char *device = arg;

    off_t size;

    size_t blocksize;

    uint32_t nbdflags;

    int fd, sock;

    int ret;

    pthread_t show_parts_thread;



    sock = unix_socket_outgoing(sockpath);

    if (sock < 0) {

        goto out;

    }



    ret = nbd_receive_negotiate(sock, NULL, &nbdflags,

                                &size, &blocksize);

    if (ret < 0) {

        goto out;

    }



    fd = open(device, O_RDWR);

    if (fd < 0) {

        /* Linux-only, we can use %m in printf.  */

        fprintf(stderr, "Failed to open %s: %m", device);

        goto out;

    }



    ret = nbd_init(fd, sock, nbdflags, size, blocksize);

    if (ret < 0) {

        goto out;

    }



    /* update partition table */

    pthread_create(&show_parts_thread, NULL, show_parts, device);



    if (verbose) {

        fprintf(stderr, "NBD device %s is now connected to %s\n",

                device, srcpath);

    } else {

        /* Close stderr so that the qemu-nbd process exits.  */

        dup2(STDOUT_FILENO, STDERR_FILENO);

    }



    ret = nbd_client(fd);

    if (ret) {

        goto out;

    }

    close(fd);

    kill(getpid(), SIGTERM);

    return (void *) EXIT_SUCCESS;



out:

    kill(getpid(), SIGTERM);

    return (void *) EXIT_FAILURE;

}
