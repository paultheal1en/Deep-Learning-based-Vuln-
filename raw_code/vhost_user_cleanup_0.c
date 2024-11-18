static void vhost_user_cleanup(NetClientState *nc)

{

    VhostUserState *s = DO_UPCAST(VhostUserState, nc, nc);



    vhost_user_stop(s);

    qemu_purge_queued_packets(nc);

}
