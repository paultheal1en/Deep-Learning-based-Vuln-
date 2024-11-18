static int do_load_save_vmstate(BDRVSheepdogState *s, uint8_t *data,

                                int64_t pos, int size, int load)

{

    bool create;

    int fd, ret = 0, remaining = size;

    unsigned int data_len;

    uint64_t vmstate_oid;

    uint32_t vdi_index;

    uint64_t offset;



    fd = connect_to_sdog(s->addr, s->port);

    if (fd < 0) {

        return fd;

    }



    while (remaining) {

        vdi_index = pos / SD_DATA_OBJ_SIZE;

        offset = pos % SD_DATA_OBJ_SIZE;



        data_len = MIN(remaining, SD_DATA_OBJ_SIZE - offset);



        vmstate_oid = vid_to_vmstate_oid(s->inode.vdi_id, vdi_index);



        create = (offset == 0);

        if (load) {

            ret = read_object(fd, (char *)data, vmstate_oid,

                              s->inode.nr_copies, data_len, offset,

                              s->cache_enabled);

        } else {

            ret = write_object(fd, (char *)data, vmstate_oid,

                               s->inode.nr_copies, data_len, offset, create,

                               s->cache_enabled);

        }



        if (ret < 0) {

            error_report("failed to save vmstate %s", strerror(errno));

            goto cleanup;

        }



        pos += data_len;

        data += data_len;

        remaining -= data_len;

    }

    ret = size;

cleanup:

    closesocket(fd);

    return ret;

}
