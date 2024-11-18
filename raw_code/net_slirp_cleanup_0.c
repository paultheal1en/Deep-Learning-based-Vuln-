static void net_slirp_cleanup(VLANClientState *vc)

{

    SlirpState *s = vc->opaque;



    slirp_cleanup(s->slirp);

    slirp_smb_cleanup(s);

    TAILQ_REMOVE(&slirp_stacks, s, entry);

    qemu_free(s);

}
