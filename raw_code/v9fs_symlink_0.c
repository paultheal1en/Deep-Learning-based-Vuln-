static void v9fs_symlink(void *opaque)

{

    V9fsPDU *pdu = opaque;

    V9fsString name;

    V9fsString symname;

    V9fsString fullname;

    V9fsFidState *dfidp;

    V9fsQID qid;

    struct stat stbuf;

    int32_t dfid;

    int err = 0;

    gid_t gid;

    size_t offset = 7;



    v9fs_string_init(&fullname);

    pdu_unmarshal(pdu, offset, "dssd", &dfid, &name, &symname, &gid);



    dfidp = get_fid(pdu->s, dfid);

    if (dfidp == NULL) {

        err = -EINVAL;

        goto out_nofid;

    }



    v9fs_string_sprintf(&fullname, "%s/%s", dfidp->path.data, name.data);

    err = v9fs_co_symlink(pdu->s, dfidp, symname.data, fullname.data, gid);

    if (err < 0) {

        goto out;

    }

    err = v9fs_co_lstat(pdu->s, &fullname, &stbuf);

    if (err < 0) {

        goto out;

    }

    stat_to_qid(&stbuf, &qid);

    offset += pdu_marshal(pdu, offset, "Q", &qid);

    err = offset;

out:

    put_fid(pdu->s, dfidp);

out_nofid:

    complete_pdu(pdu->s, pdu, err);

    v9fs_string_free(&name);

    v9fs_string_free(&symname);

    v9fs_string_free(&fullname);

}
