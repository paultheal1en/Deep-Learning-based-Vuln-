static int qiov_is_aligned(BlockDriverState *bs, QEMUIOVector *qiov)

{

    int i;



    for (i = 0; i < qiov->niov; i++) {

        if ((uintptr_t) qiov->iov[i].iov_base % bs->buffer_alignment) {

            return 0;

        }

    }



    return 1;

}
