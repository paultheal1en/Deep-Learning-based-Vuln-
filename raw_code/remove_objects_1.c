static bool remove_objects(BDRVSheepdogState *s)

{

    int fd, i = 0, nr_objs = 0;

    Error *local_err = NULL;

    int ret = 0;

    bool result = true;

    SheepdogInode *inode = &s->inode;



    fd = connect_to_sdog(s, &local_err);

    if (fd < 0) {

        error_report_err(local_err);

        return false;

    }



    nr_objs = count_data_objs(inode);

    while (i < nr_objs) {

        int start_idx, nr_filled_idx;



        while (i < nr_objs && !inode->data_vdi_id[i]) {

            i++;

        }

        start_idx = i;



        nr_filled_idx = 0;

        while (i < nr_objs && nr_filled_idx < NR_BATCHED_DISCARD) {

            if (inode->data_vdi_id[i]) {

                inode->data_vdi_id[i] = 0;

                nr_filled_idx++;

            }



            i++;

        }



        ret = write_object(fd, s->bs,

                           (char *)&inode->data_vdi_id[start_idx],

                           vid_to_vdi_oid(s->inode.vdi_id), inode->nr_copies,

                           (i - start_idx) * sizeof(uint32_t),

                           offsetof(struct SheepdogInode,

                                    data_vdi_id[start_idx]),

                           false, s->cache_flags);

        if (ret < 0) {

            error_report("failed to discard snapshot inode.");

            result = false;

            goto out;

        }

    }



out:

    closesocket(fd);

    return result;

}
