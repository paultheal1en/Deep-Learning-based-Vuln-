int qcow2_pre_write_overlap_check(BlockDriverState *bs, int ign, int64_t offset,

                                  int64_t size)

{

    int ret = qcow2_check_metadata_overlap(bs, ign, offset, size);



    if (ret < 0) {

        return ret;

    } else if (ret > 0) {

        int metadata_ol_bitnr = ffs(ret) - 1;

        char *message;



        assert(metadata_ol_bitnr < QCOW2_OL_MAX_BITNR);



        fprintf(stderr, "qcow2: Preventing invalid write on metadata (overlaps "

                "with %s); image marked as corrupt.\n",

                metadata_ol_names[metadata_ol_bitnr]);

        message = g_strdup_printf("Prevented %s overwrite",

                metadata_ol_names[metadata_ol_bitnr]);

        qapi_event_send_block_image_corrupted(bdrv_get_device_name(bs),

                                              message,

                                              true,

                                              offset,

                                              true,

                                              size,

                                              true,

                                              &error_abort);

        g_free(message);



        qcow2_mark_corrupt(bs);

        bs->drv = NULL; /* make BDS unusable */

        return -EIO;

    }



    return 0;

}
