static int sd_truncate(BlockDriverState *bs, int64_t offset)

{

    BDRVSheepdogState *s = bs->opaque;

    int ret, fd;

    unsigned int datalen;



    if (offset < s->inode.vdi_size) {

        error_report("shrinking is not supported");

        return -EINVAL;

    } else if (offset > SD_MAX_VDI_SIZE) {

        error_report("too big image size");

        return -EINVAL;

    }



    fd = connect_to_sdog(s->addr, s->port);

    if (fd < 0) {

        return fd;

    }



    /* we don't need to update entire object */

    datalen = SD_INODE_SIZE - sizeof(s->inode.data_vdi_id);

    s->inode.vdi_size = offset;

    ret = write_object(fd, (char *)&s->inode, vid_to_vdi_oid(s->inode.vdi_id),

                       s->inode.nr_copies, datalen, 0, false, s->cache_enabled);

    close(fd);



    if (ret < 0) {

        error_report("failed to update an inode.");

    }



    return ret;

}
