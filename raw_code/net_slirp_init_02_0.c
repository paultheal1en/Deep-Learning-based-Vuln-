static int net_slirp_init(VLANState *vlan, const char *model, const char *name,

                          int restricted, const char *ip)

{

    if (slirp_in_use) {

        /* slirp only supports a single instance so far */

        return -1;

    }

    if (!slirp_inited) {

        slirp_inited = 1;

        slirp_init(restricted, ip);



        while (slirp_redirs) {

            struct slirp_config_str *config = slirp_redirs;



            slirp_redirection(NULL, config->str);

            slirp_redirs = config->next;

            qemu_free(config);

        }

#ifndef _WIN32

        if (slirp_smb_export) {

            slirp_smb(slirp_smb_export);

        }

#endif

    }



    slirp_vc = qemu_new_vlan_client(vlan, model, name, NULL, slirp_receive,

                                    NULL, net_slirp_cleanup, NULL);

    slirp_vc->info_str[0] = '\0';

    slirp_in_use = 1;

    return 0;

}
