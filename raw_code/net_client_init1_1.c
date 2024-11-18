static int net_client_init1(const void *object, int is_netdev, Error **errp)

{

    union {

        const Netdev    *netdev;

        const NetLegacy *net;

    } u;

    const NetClientOptions *opts;

    const char *name;



    if (is_netdev) {

        u.netdev = object;

        opts = u.netdev->opts;

        name = u.netdev->id;



        switch (opts->kind) {

#ifdef CONFIG_SLIRP

        case NET_CLIENT_OPTIONS_KIND_USER:

#endif

        case NET_CLIENT_OPTIONS_KIND_TAP:

        case NET_CLIENT_OPTIONS_KIND_SOCKET:

#ifdef CONFIG_VDE

        case NET_CLIENT_OPTIONS_KIND_VDE:

#endif

#ifdef CONFIG_NETMAP

        case NET_CLIENT_OPTIONS_KIND_NETMAP:

#endif

#ifdef CONFIG_NET_BRIDGE

        case NET_CLIENT_OPTIONS_KIND_BRIDGE:

#endif

        case NET_CLIENT_OPTIONS_KIND_HUBPORT:

#ifdef CONFIG_VHOST_NET_USED

        case NET_CLIENT_OPTIONS_KIND_VHOST_USER:

#endif

#ifdef CONFIG_LINUX

        case NET_CLIENT_OPTIONS_KIND_L2TPV3:

#endif

            break;



        default:

            error_set(errp, QERR_INVALID_PARAMETER_VALUE, "type",

                      "a netdev backend type");

            return -1;

        }

    } else {

        u.net = object;

        opts = u.net->opts;

        /* missing optional values have been initialized to "all bits zero" */

        name = u.net->has_id ? u.net->id : u.net->name;

    }



    if (net_client_init_fun[opts->kind]) {

        NetClientState *peer = NULL;



        /* Do not add to a vlan if it's a -netdev or a nic with a netdev=

         * parameter. */

        if (!is_netdev &&

            (opts->kind != NET_CLIENT_OPTIONS_KIND_NIC ||

             !opts->nic->has_netdev)) {

            peer = net_hub_add_port(u.net->has_vlan ? u.net->vlan : 0, NULL);

        }



        if (net_client_init_fun[opts->kind](opts, name, peer) < 0) {

            /* TODO push error reporting into init() methods */

            error_set(errp, QERR_DEVICE_INIT_FAILED,

                      NetClientOptionsKind_lookup[opts->kind]);

            return -1;

        }

    }

    return 0;

}
