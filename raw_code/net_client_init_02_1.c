int net_client_init(Monitor *mon, QemuOpts *opts, int is_netdev)

{

    const char *name;

    const char *type;

    int i;



    type = qemu_opt_get(opts, "type");

    if (!type) {

        qerror_report(QERR_MISSING_PARAMETER, "type");

        return -1;

    }



    if (is_netdev) {

        if (strcmp(type, "tap") != 0 &&

#ifdef CONFIG_SLIRP

            strcmp(type, "user") != 0 &&

#endif

#ifdef CONFIG_VDE

            strcmp(type, "vde") != 0 &&

#endif

            strcmp(type, "socket") != 0) {

            qerror_report(QERR_INVALID_PARAMETER_VALUE, "type",

                          "a netdev backend type");

            return -1;

        }



        if (qemu_opt_get(opts, "vlan")) {

            qerror_report(QERR_INVALID_PARAMETER, "vlan");

            return -1;

        }

        if (qemu_opt_get(opts, "name")) {

            qerror_report(QERR_INVALID_PARAMETER, "name");

            return -1;

        }

        if (!qemu_opts_id(opts)) {

            qerror_report(QERR_MISSING_PARAMETER, "id");

            return -1;

        }

    }



    name = qemu_opts_id(opts);

    if (!name) {

        name = qemu_opt_get(opts, "name");

    }



    for (i = 0; i < NET_CLIENT_TYPE_MAX; i++) {

        if (net_client_types[i].type != NULL &&

            !strcmp(net_client_types[i].type, type)) {

            VLANState *vlan = NULL;

            int ret;



            if (qemu_opts_validate(opts, &net_client_types[i].desc[0]) == -1) {

                return -1;

            }



            /* Do not add to a vlan if it's a -netdev or a nic with a

             * netdev= parameter. */

            if (!(is_netdev ||

                  (strcmp(type, "nic") == 0 && qemu_opt_get(opts, "netdev")))) {

                vlan = qemu_find_vlan(qemu_opt_get_number(opts, "vlan", 0), 1);

            }



            ret = 0;

            if (net_client_types[i].init) {

                ret = net_client_types[i].init(opts, mon, name, vlan);

                if (ret < 0) {

                    /* TODO push error reporting into init() methods */

                    qerror_report(QERR_DEVICE_INIT_FAILED, type);

                    return -1;

                }

            }

            return ret;

        }

    }



    qerror_report(QERR_INVALID_PARAMETER_VALUE, "type",

                  "a network client type");

    return -1;

}
