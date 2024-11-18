static void v9fs_wstat_post_utime(V9fsState *s, V9fsWstatState *vs, int err)

{

    if (err < 0) {

        goto out;

    }



    if (vs->v9stat.n_gid != -1) {

        if (v9fs_do_chown(s, &vs->fidp->path, vs->v9stat.n_uid,

                    vs->v9stat.n_gid)) {

            err = -errno;

        }

    }

    v9fs_wstat_post_chown(s, vs, err);

    return;



out:

    v9fs_stat_free(&vs->v9stat);

    complete_pdu(s, vs->pdu, err);

    qemu_free(vs);

}
