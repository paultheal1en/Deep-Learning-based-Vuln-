static int sd_snapshot_delete(BlockDriverState *bs,

                              const char *snapshot_id,

                              const char *name,

                              Error **errp)

{

    unsigned long snap_id = 0;

    char snap_tag[SD_MAX_VDI_TAG_LEN];

    Error *local_err = NULL;

    int fd, ret;

    char buf[SD_MAX_VDI_LEN + SD_MAX_VDI_TAG_LEN];

    BDRVSheepdogState *s = bs->opaque;

    unsigned int wlen = SD_MAX_VDI_LEN + SD_MAX_VDI_TAG_LEN, rlen = 0;

    uint32_t vid;

    SheepdogVdiReq hdr = {

        .opcode = SD_OP_DEL_VDI,

        .data_length = wlen,

        .flags = SD_FLAG_CMD_WRITE,

    };

    SheepdogVdiRsp *rsp = (SheepdogVdiRsp *)&hdr;



    if (!remove_objects(s)) {

        return -1;

    }



    memset(buf, 0, sizeof(buf));

    memset(snap_tag, 0, sizeof(snap_tag));

    pstrcpy(buf, SD_MAX_VDI_LEN, s->name);

    ret = qemu_strtoul(snapshot_id, NULL, 10, &snap_id);

    if (ret || snap_id > UINT32_MAX) {

        error_setg(errp, "Invalid snapshot ID: %s",

                         snapshot_id ? snapshot_id : "<null>");

        return -EINVAL;

    }



    if (snap_id) {

        hdr.snapid = (uint32_t) snap_id;

    } else {

        pstrcpy(snap_tag, sizeof(snap_tag), snapshot_id);

        pstrcpy(buf + SD_MAX_VDI_LEN, SD_MAX_VDI_TAG_LEN, snap_tag);

    }



    ret = find_vdi_name(s, s->name, snap_id, snap_tag, &vid, true,

                        &local_err);

    if (ret) {

        return ret;

    }



    fd = connect_to_sdog(s, &local_err);

    if (fd < 0) {

        error_report_err(local_err);

        return -1;

    }



    ret = do_req(fd, s->bs, (SheepdogReq *)&hdr,

                 buf, &wlen, &rlen);

    closesocket(fd);

    if (ret) {

        return ret;

    }



    switch (rsp->result) {

    case SD_RES_NO_VDI:

        error_report("%s was already deleted", s->name);

    case SD_RES_SUCCESS:

        break;

    default:

        error_report("%s, %s", sd_strerror(rsp->result), s->name);

        return -1;

    }



    return ret;

}
