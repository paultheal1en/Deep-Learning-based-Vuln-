static ssize_t block_crypto_read_func(QCryptoBlock *block,

                                      void *opaque,

                                      size_t offset,

                                      uint8_t *buf,

                                      size_t buflen,

                                      Error **errp)

{

    BlockDriverState *bs = opaque;

    ssize_t ret;



    ret = bdrv_pread(bs->file, offset, buf, buflen);

    if (ret < 0) {

        error_setg_errno(errp, -ret, "Could not read encryption header");

        return ret;

    }

    return ret;

}
