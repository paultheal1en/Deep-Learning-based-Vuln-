BlockDriverAIOCB *win32_aio_submit(BlockDriverState *bs,

        QEMUWin32AIOState *aio, HANDLE hfile,

        int64_t sector_num, QEMUIOVector *qiov, int nb_sectors,

        BlockDriverCompletionFunc *cb, void *opaque, int type)

{

    struct QEMUWin32AIOCB *waiocb;

    uint64_t offset = sector_num * 512;

    DWORD rc;



    waiocb = qemu_aio_get(&win32_aio_pool, bs, cb, opaque);

    waiocb->nbytes = nb_sectors * 512;

    waiocb->qiov = qiov;

    waiocb->is_read = (type == QEMU_AIO_READ);



    if (qiov->niov > 1) {

        waiocb->buf = qemu_blockalign(bs, qiov->size);

        if (type & QEMU_AIO_WRITE) {

            char *p = waiocb->buf;

            int i;



            for (i = 0; i < qiov->niov; ++i) {

                memcpy(p, qiov->iov[i].iov_base, qiov->iov[i].iov_len);

                p += qiov->iov[i].iov_len;

            }

        }

        waiocb->is_linear = false;

    } else {

        waiocb->buf = qiov->iov[0].iov_base;

        waiocb->is_linear = true;

    }



    waiocb->ov = (OVERLAPPED) {

        .Offset = (DWORD) offset,

        .OffsetHigh = (DWORD) (offset >> 32),

        .hEvent = event_notifier_get_handle(&aio->e)

    };

    aio->count++;



    if (type & QEMU_AIO_READ) {

        rc = ReadFile(hfile, waiocb->buf, waiocb->nbytes, NULL, &waiocb->ov);

    } else {

        rc = WriteFile(hfile, waiocb->buf, waiocb->nbytes, NULL, &waiocb->ov);

    }

    if(rc == 0 && GetLastError() != ERROR_IO_PENDING) {

        goto out_dec_count;

    }

    return &waiocb->common;



out_dec_count:

    aio->count--;

    qemu_aio_release(waiocb);

    return NULL;

}
