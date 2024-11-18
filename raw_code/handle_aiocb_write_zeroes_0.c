static ssize_t handle_aiocb_write_zeroes(RawPosixAIOData *aiocb)

{

    int ret = -EOPNOTSUPP;

    BDRVRawState *s = aiocb->bs->opaque;



    if (s->has_write_zeroes == 0) {

        return -ENOTSUP;

    }



    if (aiocb->aio_type & QEMU_AIO_BLKDEV) {

#ifdef BLKZEROOUT

        do {

            uint64_t range[2] = { aiocb->aio_offset, aiocb->aio_nbytes };

            if (ioctl(aiocb->aio_fildes, BLKZEROOUT, range) == 0) {

                return 0;

            }

        } while (errno == EINTR);



        ret = -errno;

#endif

    } else {

#ifdef CONFIG_XFS

        if (s->is_xfs) {

            return xfs_write_zeroes(s, aiocb->aio_offset, aiocb->aio_nbytes);

        }

#endif

    }



    ret = translate_err(ret);

    if (ret == -ENOTSUP) {

        s->has_write_zeroes = false;

    }

    return ret;

}
