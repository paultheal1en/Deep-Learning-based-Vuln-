static int net_client_init(const char *str)

{

    const char *p;

    char *q;

    char device[64];

    char buf[1024];

    int vlan_id, ret;

    VLANState *vlan;



    p = str;

    q = device;

    while (*p != '\0' && *p != ',') {

        if ((q - device) < sizeof(device) - 1)

            *q++ = *p;

        p++;

    }

    *q = '\0';

    if (*p == ',')

        p++;

    vlan_id = 0;

    if (get_param_value(buf, sizeof(buf), "vlan", p)) {

        vlan_id = strtol(buf, NULL, 0);

    }

    vlan = qemu_find_vlan(vlan_id);

    if (!vlan) {

        fprintf(stderr, "Could not create vlan %d\n", vlan_id);

        return -1;

    }

    if (!strcmp(device, "nic")) {

        NICInfo *nd;

        uint8_t *macaddr;



        if (nb_nics >= MAX_NICS) {

            fprintf(stderr, "Too Many NICs\n");

            return -1;

        }

        nd = &nd_table[nb_nics];

        macaddr = nd->macaddr;

        macaddr[0] = 0x52;

        macaddr[1] = 0x54;

        macaddr[2] = 0x00;

        macaddr[3] = 0x12;

        macaddr[4] = 0x34;

        macaddr[5] = 0x56 + nb_nics;



        if (get_param_value(buf, sizeof(buf), "macaddr", p)) {

            if (parse_macaddr(macaddr, buf) < 0) {

                fprintf(stderr, "invalid syntax for ethernet address\n");

                return -1;

            }

        }

        if (get_param_value(buf, sizeof(buf), "model", p)) {

            nd->model = strdup(buf);

        }

        nd->vlan = vlan;

        nb_nics++;

        vlan->nb_guest_devs++;

        ret = 0;

    } else

    if (!strcmp(device, "none")) {

        /* does nothing. It is needed to signal that no network cards

           are wanted */

        ret = 0;

    } else

#ifdef CONFIG_SLIRP

    if (!strcmp(device, "user")) {

        if (get_param_value(buf, sizeof(buf), "hostname", p)) {

            pstrcpy(slirp_hostname, sizeof(slirp_hostname), buf);

        }


        ret = net_slirp_init(vlan);

    } else

#endif

#ifdef _WIN32

    if (!strcmp(device, "tap")) {

        char ifname[64];

        if (get_param_value(ifname, sizeof(ifname), "ifname", p) <= 0) {

            fprintf(stderr, "tap: no interface name\n");

            return -1;

        }


        ret = tap_win32_init(vlan, ifname);

    } else

#else

    if (!strcmp(device, "tap")) {

        char ifname[64];

        char setup_script[1024];

        int fd;

        if (get_param_value(buf, sizeof(buf), "fd", p) > 0) {

            fd = strtol(buf, NULL, 0);

            ret = -1;

            if (net_tap_fd_init(vlan, fd))

                ret = 0;

        } else {

            if (get_param_value(ifname, sizeof(ifname), "ifname", p) <= 0) {

                ifname[0] = '\0';

            }

            if (get_param_value(setup_script, sizeof(setup_script), "script", p) == 0) {

                pstrcpy(setup_script, sizeof(setup_script), DEFAULT_NETWORK_SCRIPT);

            }


            ret = net_tap_init(vlan, ifname, setup_script);

        }

    } else

#endif

    if (!strcmp(device, "socket")) {

        if (get_param_value(buf, sizeof(buf), "fd", p) > 0) {

            int fd;

            fd = strtol(buf, NULL, 0);

            ret = -1;

            if (net_socket_fd_init(vlan, fd, 1))

                ret = 0;

        } else if (get_param_value(buf, sizeof(buf), "listen", p) > 0) {

            ret = net_socket_listen_init(vlan, buf);

        } else if (get_param_value(buf, sizeof(buf), "connect", p) > 0) {

            ret = net_socket_connect_init(vlan, buf);

        } else if (get_param_value(buf, sizeof(buf), "mcast", p) > 0) {

            ret = net_socket_mcast_init(vlan, buf);

        } else {

            fprintf(stderr, "Unknown socket options: %s\n", p);

            return -1;

        }


    } else

    {

        fprintf(stderr, "Unknown network device: %s\n", device);

        return -1;

    }

    if (ret < 0) {

        fprintf(stderr, "Could not initialize device '%s'\n", device);

    }

    

    return ret;

}