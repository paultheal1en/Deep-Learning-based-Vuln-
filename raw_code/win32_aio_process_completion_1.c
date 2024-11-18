static void win32_aio_process_completion(QEMUWin32AIOState *s,

    QEMUWin32AIOCB *waiocb, DWORD count)

{

    int ret;

    s->count--;



    if (waiocb->ov.Internal != 0) {

        ret = -EIO;

    } else {

        ret = 0;

        if (count < waiocb->nbytes) {

            /* Short reads mean EOF, pad with zeros. */

            if (waiocb->is_read) {

                qemu_iovec_memset(waiocb->qiov, count, 0,

                    waiocb->qiov->size - count);

            } else {

                ret = -EINVAL;

            }

       }

    }



    if (!waiocb->is_linear) {

        if (ret == 0 && waiocb->is_read) {

            QEMUIOVector *qiov = waiocb->qiov;

            char *p = waiocb->buf;

            int i;



            for (i = 0; i < qiov->niov; ++i) {

                memcpy(qiov->iov[i].iov_base, p, qiov->iov[i].iov_len);

                p += qiov->iov[i].iov_len;

            }

            qemu_vfree(waiocb->buf);

        }

    }





    waiocb->common.cb(waiocb->common.opaque, ret);

    qemu_aio_release(waiocb);

}
