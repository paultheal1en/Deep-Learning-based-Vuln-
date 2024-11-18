static int net_tap_init(VLANState *vlan, const char *model,

                        const char *name, const char *ifname1,

                        const char *setup_script, const char *down_script)

{

    TAPState *s;

    int fd;

    char ifname[128];



    if (ifname1 != NULL)

        pstrcpy(ifname, sizeof(ifname), ifname1);

    else

        ifname[0] = '\0';

    TFR(fd = tap_open(ifname, sizeof(ifname)));

    if (fd < 0)

        return -1;



    if (!setup_script || !strcmp(setup_script, "no"))

        setup_script = "";

    if (setup_script[0] != '\0') {

	if (launch_script(setup_script, ifname, fd))

	    return -1;

    }

    s = net_tap_fd_init(vlan, model, name, fd);

    if (!s)

        return -1;

    snprintf(s->vc->info_str, sizeof(s->vc->info_str),

             "ifname=%s,script=%s,downscript=%s",

             ifname, setup_script, down_script);

    if (down_script && strcmp(down_script, "no")) {

        snprintf(s->down_script, sizeof(s->down_script), "%s", down_script);

        snprintf(s->down_script_arg, sizeof(s->down_script_arg), "%s", ifname);

    }

    return 0;

}
