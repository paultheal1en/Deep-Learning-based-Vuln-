void net_check_clients(void)

{

    VLANState *vlan;

    VLANClientState *vc;

    int has_nic, has_host_dev;



    QTAILQ_FOREACH(vlan, &vlans, next) {

        QTAILQ_FOREACH(vc, &vlan->clients, next) {

            switch (vc->info->type) {

            case NET_CLIENT_TYPE_NIC:

                has_nic = 1;

                break;

            case NET_CLIENT_TYPE_SLIRP:

            case NET_CLIENT_TYPE_TAP:

            case NET_CLIENT_TYPE_SOCKET:

            case NET_CLIENT_TYPE_VDE:

                has_host_dev = 1;

                break;

            default: ;

            }

        }

        if (has_host_dev && !has_nic)

            fprintf(stderr, "Warning: vlan %d with no nics\n", vlan->id);

        if (has_nic && !has_host_dev)

            fprintf(stderr,

                    "Warning: vlan %d is not connected to host network\n",

                    vlan->id);

    }

    QTAILQ_FOREACH(vc, &non_vlan_clients, next) {

        if (!vc->peer) {

            fprintf(stderr, "Warning: %s %s has no peer\n",

                    vc->info->type == NET_CLIENT_TYPE_NIC ? "nic" : "netdev",

                    vc->name);

        }

    }

}
