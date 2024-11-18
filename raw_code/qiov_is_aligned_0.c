static int qiov_is_aligned(QEMUIOVector *qiov)

{

    int i;



    for (i = 0; i < qiov->niov; i++) {

        if ((uintptr_t) qiov->iov[i].iov_base % BDRV_SECTOR_SIZE) {

            return 0;

        }

    }



    return 1;

}
