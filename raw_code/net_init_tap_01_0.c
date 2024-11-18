int net_init_tap(QemuOpts *opts, Monitor *mon, const char *name, VLANState *vlan)

{

    TAPState *s;

    int fd, vnet_hdr = 0;



    if (qemu_opt_get(opts, "fd")) {

        if (qemu_opt_get(opts, "ifname") ||

            qemu_opt_get(opts, "script") ||

            qemu_opt_get(opts, "downscript") ||

            qemu_opt_get(opts, "vnet_hdr")) {

            qemu_error("ifname=, script=, downscript= and vnet_hdr= is invalid with fd=\n");

            return -1;

        }



        fd = net_handle_fd_param(mon, qemu_opt_get(opts, "fd"));

        if (fd == -1) {

            return -1;

        }



        fcntl(fd, F_SETFL, O_NONBLOCK);



        vnet_hdr = tap_probe_vnet_hdr(fd);

    } else {

        if (!qemu_opt_get(opts, "script")) {

            qemu_opt_set(opts, "script", DEFAULT_NETWORK_SCRIPT);

        }



        if (!qemu_opt_get(opts, "downscript")) {

            qemu_opt_set(opts, "downscript", DEFAULT_NETWORK_DOWN_SCRIPT);

        }



        fd = net_tap_init(opts, &vnet_hdr);

        if (fd == -1) {

            return -1;

        }

    }



    s = net_tap_fd_init(vlan, "tap", name, fd, vnet_hdr);

    if (!s) {

        close(fd);

        return -1;

    }



    if (tap_set_sndbuf(s->fd, opts) < 0) {

        return -1;

    }



    if (qemu_opt_get(opts, "fd")) {

        snprintf(s->nc.info_str, sizeof(s->nc.info_str), "fd=%d", fd);

    } else {

        const char *ifname, *script, *downscript;



        ifname     = qemu_opt_get(opts, "ifname");

        script     = qemu_opt_get(opts, "script");

        downscript = qemu_opt_get(opts, "downscript");



        snprintf(s->nc.info_str, sizeof(s->nc.info_str),

                 "ifname=%s,script=%s,downscript=%s",

                 ifname, script, downscript);



        if (strcmp(downscript, "no") != 0) {

            snprintf(s->down_script, sizeof(s->down_script), "%s", downscript);

            snprintf(s->down_script_arg, sizeof(s->down_script_arg), "%s", ifname);

        }

    }



    if (vlan) {

        vlan->nb_host_devs++;

    }



    return 0;

}
