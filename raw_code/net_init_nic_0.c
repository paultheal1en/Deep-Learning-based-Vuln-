static int net_init_nic(QemuOpts *opts,

                        Monitor *mon,

                        const char *name,

                        VLANState *vlan)

{

    int idx;

    NICInfo *nd;

    const char *netdev;



    idx = nic_get_free_idx();

    if (idx == -1 || nb_nics >= MAX_NICS) {

        qemu_error("Too Many NICs\n");

        return -1;

    }



    nd = &nd_table[idx];



    memset(nd, 0, sizeof(*nd));



    if ((netdev = qemu_opt_get(opts, "netdev"))) {

        nd->netdev = qemu_find_netdev(netdev);

        if (!nd->netdev) {

            qemu_error("netdev '%s' not found\n", netdev);

            return -1;

        }

    } else {

        assert(vlan);

        nd->vlan = vlan;

    }

    if (name) {

        nd->name = qemu_strdup(name);

    }

    if (qemu_opt_get(opts, "model")) {

        nd->model = qemu_strdup(qemu_opt_get(opts, "model"));

    }

    if (qemu_opt_get(opts, "addr")) {

        nd->devaddr = qemu_strdup(qemu_opt_get(opts, "addr"));

    }



    nd->macaddr[0] = 0x52;

    nd->macaddr[1] = 0x54;

    nd->macaddr[2] = 0x00;

    nd->macaddr[3] = 0x12;

    nd->macaddr[4] = 0x34;

    nd->macaddr[5] = 0x56 + idx;



    if (qemu_opt_get(opts, "macaddr") &&

        net_parse_macaddr(nd->macaddr, qemu_opt_get(opts, "macaddr")) < 0) {

        qemu_error("invalid syntax for ethernet address\n");

        return -1;

    }



    nd->nvectors = qemu_opt_get_number(opts, "vectors", NIC_NVECTORS_UNSPECIFIED);

    if (nd->nvectors != NIC_NVECTORS_UNSPECIFIED &&

        (nd->nvectors < 0 || nd->nvectors > 0x7ffffff)) {

        qemu_error("invalid # of vectors: %d\n", nd->nvectors);

        return -1;

    }



    nd->used = 1;

    if (vlan) {

        nd->vlan->nb_guest_devs++;

    }

    nb_nics++;



    return idx;

}
