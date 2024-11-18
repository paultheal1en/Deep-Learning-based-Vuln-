int do_netdev_del(Monitor *mon, const QDict *qdict, QObject **ret_data)

{

    const char *id = qdict_get_str(qdict, "id");

    VLANClientState *vc;



    vc = qemu_find_netdev(id);

    if (!vc || vc->info->type == NET_CLIENT_TYPE_NIC) {

        qerror_report(QERR_DEVICE_NOT_FOUND, id);

        return -1;

    }

    if (vc->peer) {

        qerror_report(QERR_DEVICE_IN_USE, id);

        return -1;

    }

    qemu_del_vlan_client(vc);

    qemu_opts_del(qemu_opts_find(&qemu_netdev_opts, id));

    return 0;

}
