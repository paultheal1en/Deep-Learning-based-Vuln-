int net_init_vhost_user(const NetClientOptions *opts, const char *name,

                        NetClientState *peer, Error **errp)

{

    const NetdevVhostUserOptions *vhost_user_opts;

    CharDriverState *chr;



    assert(opts->kind == NET_CLIENT_OPTIONS_KIND_VHOST_USER);

    vhost_user_opts = opts->vhost_user;



    chr = net_vhost_parse_chardev(vhost_user_opts, errp);

    if (!chr) {

        return -1;

    }



    /* verify net frontend */

    if (qemu_opts_foreach(qemu_find_opts("device"), net_vhost_check_net,

                          (char *)name, errp)) {

        return -1;

    }





    return net_vhost_user_init(peer, "vhost_user", name, chr);

}
