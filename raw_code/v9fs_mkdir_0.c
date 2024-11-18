static void v9fs_mkdir(void *opaque)

{

    V9fsPDU *pdu = opaque;

    size_t offset = 7;

    int32_t fid;

    struct stat stbuf;

    V9fsQID qid;

    V9fsString name;

    V9fsFidState *fidp;

    gid_t gid;

    int mode;

    int err = 0;



    pdu_unmarshal(pdu, offset, "dsdd", &fid, &name, &mode, &gid);



    trace_v9fs_mkdir(pdu->tag, pdu->id, fid, name.data, mode, gid);



    fidp = get_fid(pdu, fid);

    if (fidp == NULL) {

        err = -ENOENT;

        goto out_nofid;

    }

    err = v9fs_co_mkdir(pdu, fidp, &name, mode, fidp->uid, gid, &stbuf);

    if (err < 0) {

        goto out;

    }

    stat_to_qid(&stbuf, &qid);

    offset += pdu_marshal(pdu, offset, "Q", &qid);

    err = offset;

    trace_v9fs_mkdir_return(pdu->tag, pdu->id,

                            qid.type, qid.version, qid.path, err);

out:

    put_fid(pdu, fidp);

out_nofid:

    complete_pdu(pdu->s, pdu, err);

    v9fs_string_free(&name);

}
