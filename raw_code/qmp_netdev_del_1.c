void qmp_netdev_del(const char *id, Error **errp)

{

    NetClientState *nc;



    nc = qemu_find_netdev(id);

    if (!nc) {

        error_set(errp, QERR_DEVICE_NOT_FOUND, id);

        return;

    }



    qemu_del_net_client(nc);

    qemu_opts_del(qemu_opts_find(qemu_find_opts_err("netdev", errp), id));

}
