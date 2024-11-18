static int coroutine_fn sd_co_flush_to_disk(BlockDriverState *bs)

{

    BDRVSheepdogState *s = bs->opaque;

    SheepdogObjReq hdr = { 0 };

    SheepdogObjRsp *rsp = (SheepdogObjRsp *)&hdr;

    SheepdogInode *inode = &s->inode;

    int ret;

    unsigned int wlen = 0, rlen = 0;



    if (s->cache_flags != SD_FLAG_CMD_CACHE) {

        return 0;

    }



    hdr.opcode = SD_OP_FLUSH_VDI;

    hdr.oid = vid_to_vdi_oid(inode->vdi_id);



    ret = do_req(s->flush_fd, (SheepdogReq *)&hdr, NULL, &wlen, &rlen);

    if (ret) {

        error_report("failed to send a request to the sheep");

        return ret;

    }



    if (rsp->result == SD_RES_INVALID_PARMS) {

        dprintf("disable write cache since the server doesn't support it\n");



        s->cache_flags = SD_FLAG_CMD_DIRECT;

        closesocket(s->flush_fd);

        return 0;

    }



    if (rsp->result != SD_RES_SUCCESS) {

        error_report("%s", sd_strerror(rsp->result));

        return -EIO;

    }



    return 0;

}
