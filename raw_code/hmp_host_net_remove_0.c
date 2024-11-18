void hmp_host_net_remove(Monitor *mon, const QDict *qdict)

{

    NetClientState *nc;

    int vlan_id = qdict_get_int(qdict, "vlan_id");

    const char *device = qdict_get_str(qdict, "device");



    nc = net_hub_find_client_by_name(vlan_id, device);

    if (!nc) {

        error_report("Host network device '%s' on hub '%d' not found",

                     device, vlan_id);

        return;

    }

    if (!net_host_check_device(nc->model)) {

        error_report("invalid host network device '%s'", device);

        return;

    }



    qemu_del_net_client(nc->peer);

    qemu_del_net_client(nc);

}
