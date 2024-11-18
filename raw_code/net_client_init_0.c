int net_client_init(Monitor *mon, const char *device, const char *p)

{

    static const char * const fd_params[] = {

        "vlan", "name", "fd", NULL

    };

    char buf[1024];

    int vlan_id, ret;

    VLANState *vlan;

    char *name = NULL;



    vlan_id = 0;

    if (get_param_value(buf, sizeof(buf), "vlan", p)) {

        vlan_id = strtol(buf, NULL, 0);

    }

    vlan = qemu_find_vlan(vlan_id);



    if (get_param_value(buf, sizeof(buf), "name", p)) {

        name = qemu_strdup(buf);

    }

    if (!strcmp(device, "nic")) {

        static const char * const nic_params[] = {

            "vlan", "name", "macaddr", "model", "addr", NULL

        };

        NICInfo *nd;

        uint8_t *macaddr;

        int idx = nic_get_free_idx();



        if (check_params(buf, sizeof(buf), nic_params, p) < 0) {

            config_error(mon, "invalid parameter '%s' in '%s'\n", buf, p);

            ret = -1;

            goto out;

        }

        if (idx == -1 || nb_nics >= MAX_NICS) {

            config_error(mon, "Too Many NICs\n");

            ret = -1;

            goto out;

        }

        nd = &nd_table[idx];

        macaddr = nd->macaddr;

        macaddr[0] = 0x52;

        macaddr[1] = 0x54;

        macaddr[2] = 0x00;

        macaddr[3] = 0x12;

        macaddr[4] = 0x34;

        macaddr[5] = 0x56 + idx;



        if (get_param_value(buf, sizeof(buf), "macaddr", p)) {

            if (parse_macaddr(macaddr, buf) < 0) {

                config_error(mon, "invalid syntax for ethernet address\n");

                ret = -1;

                goto out;

            }

        }

        if (get_param_value(buf, sizeof(buf), "model", p)) {

            nd->model = strdup(buf);

        }

        if (get_param_value(buf, sizeof(buf), "addr", p)) {

            nd->devaddr = strdup(buf);

        }

        nd->vlan = vlan;

        nd->name = name;

        nd->used = 1;

        name = NULL;

        nb_nics++;

        vlan->nb_guest_devs++;

        ret = idx;

    } else

    if (!strcmp(device, "none")) {

        if (*p != '\0') {

            config_error(mon, "'none' takes no parameters\n");

            ret = -1;

            goto out;

        }

        /* does nothing. It is needed to signal that no network cards

           are wanted */

        ret = 0;

    } else

#ifdef CONFIG_SLIRP

    if (!strcmp(device, "user")) {

        static const char * const slirp_params[] = {

            "vlan", "name", "hostname", "restrict", "ip", NULL

        };

        int restricted = 0;

        char *ip = NULL;



        if (check_params(buf, sizeof(buf), slirp_params, p) < 0) {

            config_error(mon, "invalid parameter '%s' in '%s'\n", buf, p);

            ret = -1;

            goto out;

        }

        if (get_param_value(buf, sizeof(buf), "hostname", p)) {

            pstrcpy(slirp_hostname, sizeof(slirp_hostname), buf);

        }

        if (get_param_value(buf, sizeof(buf), "restrict", p)) {

            restricted = (buf[0] == 'y') ? 1 : 0;

        }

        if (get_param_value(buf, sizeof(buf), "ip", p)) {

            ip = qemu_strdup(buf);

        }

        vlan->nb_host_devs++;

        ret = net_slirp_init(vlan, device, name, restricted, ip);

        qemu_free(ip);

    } else if (!strcmp(device, "channel")) {

        long port;

        char name[20], *devname;

        struct VMChannel *vmc;



        port = strtol(p, &devname, 10);

        devname++;

        if (port < 1 || port > 65535) {

            config_error(mon, "vmchannel wrong port number\n");

            ret = -1;

            goto out;

        }

        vmc = malloc(sizeof(struct VMChannel));

        snprintf(name, 20, "vmchannel%ld", port);

        vmc->hd = qemu_chr_open(name, devname, NULL);

        if (!vmc->hd) {

            config_error(mon, "could not open vmchannel device '%s'\n",

                         devname);

            ret = -1;

            goto out;

        }

        vmc->port = port;

        slirp_add_exec(3, vmc->hd, 4, port);

        qemu_chr_add_handlers(vmc->hd, vmchannel_can_read, vmchannel_read,

                NULL, vmc);

        ret = 0;

    } else

#endif

#ifdef _WIN32

    if (!strcmp(device, "tap")) {

        static const char * const tap_params[] = {

            "vlan", "name", "ifname", NULL

        };

        char ifname[64];



        if (check_params(buf, sizeof(buf), tap_params, p) < 0) {

            config_error(mon, "invalid parameter '%s' in '%s'\n", buf, p);

            ret = -1;

            goto out;

        }

        if (get_param_value(ifname, sizeof(ifname), "ifname", p) <= 0) {

            config_error(mon, "tap: no interface name\n");

            ret = -1;

            goto out;

        }

        vlan->nb_host_devs++;

        ret = tap_win32_init(vlan, device, name, ifname);

    } else

#elif defined (_AIX)

#else

    if (!strcmp(device, "tap")) {

        char ifname[64], chkbuf[64];

        char setup_script[1024], down_script[1024];

        TAPState *s;

        int fd;

        vlan->nb_host_devs++;

        if (get_param_value(buf, sizeof(buf), "fd", p) > 0) {

            if (check_params(chkbuf, sizeof(chkbuf), fd_params, p) < 0) {

                config_error(mon, "invalid parameter '%s' in '%s'\n", chkbuf, p);

                ret = -1;

                goto out;

            }

            fd = strtol(buf, NULL, 0);

            fcntl(fd, F_SETFL, O_NONBLOCK);

            s = net_tap_fd_init(vlan, device, name, fd);

        } else {

            static const char * const tap_params[] = {

                "vlan", "name", "ifname", "script", "downscript", NULL

            };

            if (check_params(chkbuf, sizeof(chkbuf), tap_params, p) < 0) {

                config_error(mon, "invalid parameter '%s' in '%s'\n", chkbuf, p);

                ret = -1;

                goto out;

            }

            if (get_param_value(ifname, sizeof(ifname), "ifname", p) <= 0) {

                ifname[0] = '\0';

            }

            if (get_param_value(setup_script, sizeof(setup_script), "script", p) == 0) {

                pstrcpy(setup_script, sizeof(setup_script), DEFAULT_NETWORK_SCRIPT);

            }

            if (get_param_value(down_script, sizeof(down_script), "downscript", p) == 0) {

                pstrcpy(down_script, sizeof(down_script), DEFAULT_NETWORK_DOWN_SCRIPT);

            }

            s = net_tap_init(vlan, device, name, ifname, setup_script, down_script);

        }

        if (s != NULL) {

            ret = 0;

        } else {

            ret = -1;

        }

    } else

#endif

    if (!strcmp(device, "socket")) {

        char chkbuf[64];

        if (get_param_value(buf, sizeof(buf), "fd", p) > 0) {

            int fd;

            if (check_params(chkbuf, sizeof(chkbuf), fd_params, p) < 0) {

                config_error(mon, "invalid parameter '%s' in '%s'\n", chkbuf, p);

                ret = -1;

                goto out;

            }

            fd = strtol(buf, NULL, 0);

            ret = -1;

            if (net_socket_fd_init(vlan, device, name, fd, 1))

                ret = 0;

        } else if (get_param_value(buf, sizeof(buf), "listen", p) > 0) {

            static const char * const listen_params[] = {

                "vlan", "name", "listen", NULL

            };

            if (check_params(chkbuf, sizeof(chkbuf), listen_params, p) < 0) {

                config_error(mon, "invalid parameter '%s' in '%s'\n", chkbuf, p);

                ret = -1;

                goto out;

            }

            ret = net_socket_listen_init(vlan, device, name, buf);

        } else if (get_param_value(buf, sizeof(buf), "connect", p) > 0) {

            static const char * const connect_params[] = {

                "vlan", "name", "connect", NULL

            };

            if (check_params(chkbuf, sizeof(chkbuf), connect_params, p) < 0) {

                config_error(mon, "invalid parameter '%s' in '%s'\n", chkbuf, p);

                ret = -1;

                goto out;

            }

            ret = net_socket_connect_init(vlan, device, name, buf);

        } else if (get_param_value(buf, sizeof(buf), "mcast", p) > 0) {

            static const char * const mcast_params[] = {

                "vlan", "name", "mcast", NULL

            };

            if (check_params(chkbuf, sizeof(chkbuf), mcast_params, p) < 0) {

                config_error(mon, "invalid parameter '%s' in '%s'\n", chkbuf, p);

                ret = -1;

                goto out;

            }

            ret = net_socket_mcast_init(vlan, device, name, buf);

        } else {

            config_error(mon, "Unknown socket options: %s\n", p);

            ret = -1;

            goto out;

        }

        vlan->nb_host_devs++;

    } else

#ifdef CONFIG_VDE

    if (!strcmp(device, "vde")) {

        static const char * const vde_params[] = {

            "vlan", "name", "sock", "port", "group", "mode", NULL

        };

        char vde_sock[1024], vde_group[512];

	int vde_port, vde_mode;



        if (check_params(buf, sizeof(buf), vde_params, p) < 0) {

            config_error(mon, "invalid parameter '%s' in '%s'\n", buf, p);

            ret = -1;

            goto out;

        }

        vlan->nb_host_devs++;

        if (get_param_value(vde_sock, sizeof(vde_sock), "sock", p) <= 0) {

	    vde_sock[0] = '\0';

	}

	if (get_param_value(buf, sizeof(buf), "port", p) > 0) {

	    vde_port = strtol(buf, NULL, 10);

	} else {

	    vde_port = 0;

	}

	if (get_param_value(vde_group, sizeof(vde_group), "group", p) <= 0) {

	    vde_group[0] = '\0';

	}

	if (get_param_value(buf, sizeof(buf), "mode", p) > 0) {

	    vde_mode = strtol(buf, NULL, 8);

	} else {

	    vde_mode = 0700;

	}

	ret = net_vde_init(vlan, device, name, vde_sock, vde_port, vde_group, vde_mode);

    } else

#endif

    if (!strcmp(device, "dump")) {

        int len = 65536;



        if (get_param_value(buf, sizeof(buf), "len", p) > 0) {

            len = strtol(buf, NULL, 0);

        }

        if (!get_param_value(buf, sizeof(buf), "file", p)) {

            snprintf(buf, sizeof(buf), "qemu-vlan%d.pcap", vlan_id);

        }

        ret = net_dump_init(mon, vlan, device, name, buf, len);

    } else {

        config_error(mon, "Unknown network device: %s\n", device);

        ret = -1;

        goto out;

    }

    if (ret < 0) {

        config_error(mon, "Could not initialize device '%s'\n", device);

    }

out:

    qemu_free(name);

    return ret;

}
