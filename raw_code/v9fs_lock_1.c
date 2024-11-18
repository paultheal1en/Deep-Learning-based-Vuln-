static void v9fs_lock(void *opaque)
{
    int8_t status;
    V9fsFlock *flock;
    size_t offset = 7;
    struct stat stbuf;
    V9fsFidState *fidp;
    int32_t fid, err = 0;
    V9fsPDU *pdu = opaque;
    V9fsState *s = pdu->s;
    flock = g_malloc(sizeof(*flock));
    pdu_unmarshal(pdu, offset, "dbdqqds", &fid, &flock->type,
                  &flock->flags, &flock->start, &flock->length,
                  &flock->proc_id, &flock->client_id);
    status = P9_LOCK_ERROR;
    /* We support only block flag now (that too ignored currently) */
    if (flock->flags & ~P9_LOCK_FLAGS_BLOCK) {
        err = -EINVAL;
        goto out_nofid;
    }
    fidp = get_fid(pdu, fid);
    if (fidp == NULL) {
        err = -ENOENT;
        goto out_nofid;
    }
    err = v9fs_co_fstat(pdu, fidp->fs.fd, &stbuf);
    if (err < 0) {
        goto out;
    }
    status = P9_LOCK_SUCCESS;
out:
    put_fid(pdu, fidp);
out_nofid:
    err = offset;
    err += pdu_marshal(pdu, offset, "b", status);
    trace_v9fs_lock_return(pdu->tag, pdu->id, status);
    complete_pdu(s, pdu, err);
    v9fs_string_free(&flock->client_id);
    g_free(flock);
}