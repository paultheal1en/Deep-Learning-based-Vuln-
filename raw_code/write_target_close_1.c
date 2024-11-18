static void write_target_close(BlockDriverState *bs) {

    BDRVVVFATState* s = *((BDRVVVFATState**) bs->opaque);

    bdrv_delete(s->qcow);

    free(s->qcow_filename);

}
