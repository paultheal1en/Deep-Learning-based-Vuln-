static int net_vhost_user_init(NetClientState *peer, const char *device,

                               const char *name, CharDriverState *chr)

{

    NetClientState *nc;

    VhostUserState *s;



    nc = qemu_new_net_client(&net_vhost_user_info, peer, device, name);



    snprintf(nc->info_str, sizeof(nc->info_str), "vhost-user to %s",

             chr->label);



    s = DO_UPCAST(VhostUserState, nc, nc);



    /* We don't provide a receive callback */

    s->nc.receive_disabled = 1;

    s->chr = chr;



    qemu_chr_add_handlers(s->chr, NULL, NULL, net_vhost_user_event, s);



    return 0;

}
