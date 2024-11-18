int vhdx_parse_log(BlockDriverState *bs, BDRVVHDXState *s, bool *flushed)

{

    int ret = 0;

    VHDXHeader *hdr;

    VHDXLogSequence logs = { 0 };



    hdr = s->headers[s->curr_header];



    *flushed = false;



    /* s->log.hdr is freed in vhdx_close() */

    if (s->log.hdr == NULL) {

        s->log.hdr = qemu_blockalign(bs, sizeof(VHDXLogEntryHeader));

    }



    s->log.offset = hdr->log_offset;

    s->log.length = hdr->log_length;



    if (s->log.offset < VHDX_LOG_MIN_SIZE ||

        s->log.offset % VHDX_LOG_MIN_SIZE) {

        ret = -EINVAL;

        goto exit;

    }



    /* per spec, only log version of 0 is supported */

    if (hdr->log_version != 0) {

        ret = -EINVAL;

        goto exit;

    }



    /* If either the log guid, or log length is zero,

     * then a replay log is not present */

    if (guid_eq(hdr->log_guid, zero_guid)) {

        goto exit;

    }



    if (hdr->log_length == 0) {

        goto exit;

    }



    if (hdr->log_length % VHDX_LOG_MIN_SIZE) {

        ret = -EINVAL;

        goto exit;

    }





    /* The log is present, we need to find if and where there is an active

     * sequence of valid entries present in the log.  */



    ret = vhdx_log_search(bs, s, &logs);

    if (ret < 0) {

        goto exit;

    }



    if (logs.valid) {

        /* now flush the log */

        ret = vhdx_log_flush(bs, s, &logs);

        if (ret < 0) {

            goto exit;

        }

        *flushed = true;

    }





exit:

    return ret;

}
