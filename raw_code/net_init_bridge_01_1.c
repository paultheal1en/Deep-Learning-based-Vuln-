int net_init_bridge(QemuOpts *opts, const char *name, VLANState *vlan)

{

    TAPState *s;

    int fd, vnet_hdr;



    if (!qemu_opt_get(opts, "br")) {

        qemu_opt_set(opts, "br", DEFAULT_BRIDGE_INTERFACE);

    }

    if (!qemu_opt_get(opts, "helper")) {

        qemu_opt_set(opts, "helper", DEFAULT_BRIDGE_HELPER);

    }



    fd = net_bridge_run_helper(qemu_opt_get(opts, "helper"),

                               qemu_opt_get(opts, "br"));

    if (fd == -1) {

        return -1;

    }



    fcntl(fd, F_SETFL, O_NONBLOCK);



    vnet_hdr = tap_probe_vnet_hdr(fd);



    s = net_tap_fd_init(vlan, "bridge", name, fd, vnet_hdr);

    if (!s) {

        close(fd);

        return -1;

    }



    snprintf(s->nc.info_str, sizeof(s->nc.info_str), "helper=%s,br=%s",

             qemu_opt_get(opts, "helper"), qemu_opt_get(opts, "br"));



    return 0;

}
