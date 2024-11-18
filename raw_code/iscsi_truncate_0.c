static int iscsi_truncate(BlockDriverState *bs, int64_t offset)

{

    IscsiLun *iscsilun = bs->opaque;

    Error *local_err = NULL;



    if (iscsilun->type != TYPE_DISK) {

        return -ENOTSUP;

    }



    iscsi_readcapacity_sync(iscsilun, &local_err);

    if (local_err != NULL) {

        error_free(local_err);

        return -EIO;

    }



    if (offset > iscsi_getlength(bs)) {

        return -EINVAL;

    }



    if (iscsilun->allocationmap != NULL) {

        g_free(iscsilun->allocationmap);

        iscsilun->allocationmap = iscsi_allocationmap_init(iscsilun);

    }



    return 0;

}
