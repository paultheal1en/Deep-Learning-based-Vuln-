static void net_slirp_cleanup(NetClientState *nc)

{

    SlirpState *s = DO_UPCAST(SlirpState, nc, nc);



    slirp_cleanup(s->slirp);

    qemu_remove_exit_notifier(&s->exit_notifier);

    slirp_smb_cleanup(s);

    QTAILQ_REMOVE(&slirp_stacks, s, entry);

}
