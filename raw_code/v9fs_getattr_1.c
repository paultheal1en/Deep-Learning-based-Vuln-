static void v9fs_getattr(void *opaque)

{

    int32_t fid;

    size_t offset = 7;

    ssize_t retval = 0;

    struct stat stbuf;

    V9fsFidState *fidp;

    uint64_t request_mask;

    V9fsStatDotl v9stat_dotl;

    V9fsPDU *pdu = opaque;

    V9fsState *s = pdu->s;



    pdu_unmarshal(pdu, offset, "dq", &fid, &request_mask);




    fidp = get_fid(pdu, fid);

    if (fidp == NULL) {

        retval = -ENOENT;

        goto out_nofid;

    }

    /*

     * Currently we only support BASIC fields in stat, so there is no

     * need to look at request_mask.

     */

    retval = v9fs_co_lstat(pdu, &fidp->path, &stbuf);

    if (retval < 0) {

        goto out;

    }

    stat_to_v9stat_dotl(s, &stbuf, &v9stat_dotl);



    /*  fill st_gen if requested and supported by underlying fs */

    if (request_mask & P9_STATS_GEN) {

        retval = v9fs_co_st_gen(pdu, &fidp->path, stbuf.st_mode, &v9stat_dotl);

        if (retval < 0) {

            goto out;

        }

        v9stat_dotl.st_result_mask |= P9_STATS_GEN;

    }

    retval = offset;

    retval += pdu_marshal(pdu, offset, "A", &v9stat_dotl);

out:

    put_fid(pdu, fidp);

out_nofid:

    trace_v9fs_getattr_return(pdu->tag, pdu->id, v9stat_dotl.st_result_mask,

                              v9stat_dotl.st_mode, v9stat_dotl.st_uid,

                              v9stat_dotl.st_gid);



    complete_pdu(s, pdu, retval);

}