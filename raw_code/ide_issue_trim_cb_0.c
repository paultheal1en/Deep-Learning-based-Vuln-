static void ide_issue_trim_cb(void *opaque, int ret)

{

    TrimAIOCB *iocb = opaque;

    if (ret >= 0) {

        while (iocb->j < iocb->qiov->niov) {

            int j = iocb->j;

            while (++iocb->i < iocb->qiov->iov[j].iov_len / 8) {

                int i = iocb->i;

                uint64_t *buffer = iocb->qiov->iov[j].iov_base;



                /* 6-byte LBA + 2-byte range per entry */

                uint64_t entry = le64_to_cpu(buffer[i]);

                uint64_t sector = entry & 0x0000ffffffffffffULL;

                uint16_t count = entry >> 48;



                if (count == 0) {

                    continue;

                }



                /* Got an entry! Submit and exit.  */

                iocb->aiocb = blk_aio_pdiscard(iocb->blk,

                                               sector << BDRV_SECTOR_BITS,

                                               count << BDRV_SECTOR_BITS,

                                               ide_issue_trim_cb, opaque);

                return;

            }



            iocb->j++;

            iocb->i = -1;

        }

    } else {

        iocb->ret = ret;

    }



    iocb->aiocb = NULL;

    if (iocb->bh) {

        qemu_bh_schedule(iocb->bh);

    }

}
