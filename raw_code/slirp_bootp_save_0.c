static void slirp_bootp_save(QEMUFile *f, Slirp *slirp)

{

    int i;



    for (i = 0; i < NB_BOOTP_CLIENTS; i++) {

        qemu_put_be16(f, slirp->bootp_clients[i].allocated);

        qemu_put_buffer(f, slirp->bootp_clients[i].macaddr, 6);

    }

}
