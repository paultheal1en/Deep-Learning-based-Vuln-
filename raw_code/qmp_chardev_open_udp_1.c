static CharDriverState *qmp_chardev_open_udp(ChardevUdp *udp,

                                             Error **errp)

{

    int fd;



    fd = socket_dgram(udp->remote, udp->local, errp);

    if (error_is_set(errp)) {

        return NULL;

    }

    return qemu_chr_open_udp_fd(fd);

}
