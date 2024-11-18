static void tcp_accept_incoming_migration(void *opaque)

{

    struct sockaddr_in addr;

    socklen_t addrlen = sizeof(addr);

    int s = (intptr_t)opaque;

    QEMUFile *f;

    int c;



    do {

        c = qemu_accept(s, (struct sockaddr *)&addr, &addrlen);

    } while (c == -1 && socket_error() == EINTR);

    qemu_set_fd_handler2(s, NULL, NULL, NULL, NULL);

    closesocket(s);



    DPRINTF("accepted migration\n");



    if (c == -1) {

        fprintf(stderr, "could not accept migration connection\n");

        goto out;

    }



    f = qemu_fopen_socket(c, "rb");

    if (f == NULL) {

        fprintf(stderr, "could not qemu_fopen socket\n");

        goto out;

    }



    process_incoming_migration(f);

    return;



out:

    closesocket(c);

}
