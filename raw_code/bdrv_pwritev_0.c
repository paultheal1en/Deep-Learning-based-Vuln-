int bdrv_pwritev(BlockDriverState *bs, int64_t offset, QEMUIOVector *qiov)

{

    int ret;



    ret = bdrv_prwv_co(bs, offset, qiov, true, 0);

    if (ret < 0) {

        return ret;

    }



    return qiov->size;

}
