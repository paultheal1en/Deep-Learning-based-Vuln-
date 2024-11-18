int qcow2_pre_write_overlap_check(BlockDriverState *bs, int ign, int64_t offset,

                                  int64_t size)

{

    int ret = qcow2_check_metadata_overlap(bs, ign, offset, size);



    if (ret < 0) {

        return ret;

    } else if (ret > 0) {

        int metadata_ol_bitnr = ffs(ret) - 1;

        assert(metadata_ol_bitnr < QCOW2_OL_MAX_BITNR);



        qcow2_signal_corruption(bs, true, offset, size, "Preventing invalid "

                                "write on metadata (overlaps with %s)",

                                metadata_ol_names[metadata_ol_bitnr]);

        return -EIO;

    }



    return 0;

}
