void slirp_cleanup(Slirp *slirp)

{

    TAILQ_REMOVE(&slirp_instances, slirp, entry);



    unregister_savevm("slirp", slirp);



    qemu_free(slirp->tftp_prefix);

    qemu_free(slirp->bootp_filename);

    qemu_free(slirp);

}
