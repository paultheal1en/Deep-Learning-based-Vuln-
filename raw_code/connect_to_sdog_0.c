static int connect_to_sdog(BDRVSheepdogState *s, Error **errp)

{

    int fd;



    fd = socket_connect(s->addr, NULL, NULL, errp);



    if (s->addr->type == SOCKET_ADDRESS_KIND_INET && fd >= 0) {

        int ret = socket_set_nodelay(fd);

        if (ret < 0) {

            error_report("%s", strerror(errno));

        }

    }



    if (fd >= 0) {

        qemu_set_nonblock(fd);

    } else {

        fd = -EIO;

    }



    return fd;

}
