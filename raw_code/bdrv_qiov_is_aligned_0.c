bool bdrv_qiov_is_aligned(BlockDriverState *bs, QEMUIOVector *qiov)

{

    int i;

    size_t alignment = bdrv_opt_mem_align(bs);



    for (i = 0; i < qiov->niov; i++) {

        if ((uintptr_t) qiov->iov[i].iov_base % alignment) {

            return false;

        }

        if (qiov->iov[i].iov_len % alignment) {

            return false;

        }

    }



    return true;

}
