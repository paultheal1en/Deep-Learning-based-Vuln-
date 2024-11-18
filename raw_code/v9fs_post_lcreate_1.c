static void v9fs_post_lcreate(V9fsState *s, V9fsLcreateState *vs, int err)

{

    if (err == 0) {

        v9fs_string_copy(&vs->fidp->path, &vs->fullname);

        stat_to_qid(&vs->stbuf, &vs->qid);

        vs->offset += pdu_marshal(vs->pdu, vs->offset, "Qd", &vs->qid,

                &vs->iounit);

        err = vs->offset;

    } else {

        vs->fidp->fid_type = P9_FID_NONE;

        close(vs->fidp->fs.fd);

        err = -errno;

    }



    complete_pdu(s, vs->pdu, err);

    v9fs_string_free(&vs->name);

    v9fs_string_free(&vs->fullname);

    qemu_free(vs);

}
