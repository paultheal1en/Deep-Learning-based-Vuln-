static void v9fs_rename(void *opaque)

{

    int32_t fid;

    ssize_t err = 0;

    size_t offset = 7;

    V9fsString name;

    int32_t newdirfid;

    V9fsFidState *fidp;

    V9fsPDU *pdu = opaque;

    V9fsState *s = pdu->s;



    pdu_unmarshal(pdu, offset, "dds", &fid, &newdirfid, &name);



    fidp = get_fid(pdu, fid);

    if (fidp == NULL) {

        err = -ENOENT;

        goto out_nofid;

    }

    BUG_ON(fidp->fid_type != P9_FID_NONE);

    /* if fs driver is not path based, return EOPNOTSUPP */

    if (!(pdu->s->ctx.export_flags & V9FS_PATHNAME_FSCONTEXT)) {

        err = -EOPNOTSUPP;

        goto out;

    }

    v9fs_path_write_lock(s);

    err = v9fs_complete_rename(pdu, fidp, newdirfid, &name);

    v9fs_path_unlock(s);

    if (!err) {

        err = offset;

    }

out:

    put_fid(pdu, fidp);

out_nofid:

    complete_pdu(s, pdu, err);

    v9fs_string_free(&name);

}
