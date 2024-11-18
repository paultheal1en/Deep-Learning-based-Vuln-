int net_init_slirp(QemuOpts *opts,

                   Monitor *mon,

                   const char *name,

                   VLANState *vlan)

{

    struct slirp_config_str *config;

    const char *vhost;

    const char *vhostname;

    const char *vdhcp_start;

    const char *vnamesrv;

    const char *tftp_export;

    const char *bootfile;

    const char *smb_export;

    const char *vsmbsrv;

    char *vnet = NULL;

    int restricted = 0;

    int ret;



    vhost       = qemu_opt_get(opts, "host");

    vhostname   = qemu_opt_get(opts, "hostname");

    vdhcp_start = qemu_opt_get(opts, "dhcpstart");

    vnamesrv    = qemu_opt_get(opts, "dns");

    tftp_export = qemu_opt_get(opts, "tftp");

    bootfile    = qemu_opt_get(opts, "bootfile");

    smb_export  = qemu_opt_get(opts, "smb");

    vsmbsrv     = qemu_opt_get(opts, "smbserver");



    if (qemu_opt_get(opts, "ip")) {

        const char *ip = qemu_opt_get(opts, "ip");

        int l = strlen(ip) + strlen("/24") + 1;



        vnet = qemu_malloc(l);



        /* emulate legacy ip= parameter */

        pstrcpy(vnet, l, ip);

        pstrcat(vnet, l, "/24");

    }



    if (qemu_opt_get(opts, "net")) {

        if (vnet) {

            qemu_free(vnet);

        }

        vnet = qemu_strdup(qemu_opt_get(opts, "net"));

    }



    if (qemu_opt_get(opts, "restrict") &&

        qemu_opt_get(opts, "restrict")[0] == 'y') {

        restricted = 1;

    }



    qemu_opt_foreach(opts, net_init_slirp_configs, NULL, 0);



    ret = net_slirp_init(vlan, "user", name, restricted, vnet, vhost,

                         vhostname, tftp_export, bootfile, vdhcp_start,

                         vnamesrv, smb_export, vsmbsrv);



    while (slirp_configs) {

        config = slirp_configs;

        slirp_configs = config->next;

        qemu_free(config);

    }



    if (ret != -1 && vlan) {

        vlan->nb_host_devs++;

    }



    qemu_free(vnet);



    return ret;

}
