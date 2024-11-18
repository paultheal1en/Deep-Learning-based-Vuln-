static void v9fs_wstat_post_rename(V9fsState *s, V9fsWstatState *vs, int err)

{

    if (err < 0) {

        goto out;

    }



    if (vs->v9stat.name.size != 0) {

        v9fs_string_free(&vs->nname);

    }



    if (vs->v9stat.length != -1) {

        if (v9fs_do_truncate(s, &vs->fidp->path, vs->v9stat.length) < 0) {

            err = -errno;

        }

    }

    v9fs_wstat_post_truncate(s, vs, err);

    return;



out:

    v9fs_stat_free(&vs->v9stat);

    complete_pdu(s, vs->pdu, err);

    qemu_free(vs);

}
