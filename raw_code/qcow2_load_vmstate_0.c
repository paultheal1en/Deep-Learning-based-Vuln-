static int qcow2_load_vmstate(BlockDriverState *bs, QEMUIOVector *qiov,

                              int64_t pos)

{

    BDRVQcow2State *s = bs->opaque;

    bool zero_beyond_eof = bs->zero_beyond_eof;

    int ret;



    BLKDBG_EVENT(bs->file, BLKDBG_VMSTATE_LOAD);

    bs->zero_beyond_eof = false;

    ret = bdrv_preadv(bs, qcow2_vm_state_offset(s) + pos, qiov);

    bs->zero_beyond_eof = zero_beyond_eof;



    return ret;

}
