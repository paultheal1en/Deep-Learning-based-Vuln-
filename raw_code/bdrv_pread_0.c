int bdrv_pread(BlockDriverState *bs, int64_t offset, void *buf, int bytes)

{

    QEMUIOVector qiov;

    struct iovec iov = {

        .iov_base = (void *)buf,

        .iov_len = bytes,

    };

    int ret;



    if (bytes < 0) {

        return -EINVAL;

    }



    qemu_iovec_init_external(&qiov, &iov, 1);

    ret = bdrv_prwv_co(bs, offset, &qiov, false, 0);

    if (ret < 0) {

        return ret;

    }



    return bytes;

}
