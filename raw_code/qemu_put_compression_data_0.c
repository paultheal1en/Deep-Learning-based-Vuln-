ssize_t qemu_put_compression_data(QEMUFile *f, const uint8_t *p, size_t size,

                                  int level)

{

    ssize_t blen = IO_BUF_SIZE - f->buf_index - sizeof(int32_t);



    if (blen < compressBound(size)) {

        return 0;

    }

    if (compress2(f->buf + f->buf_index + sizeof(int32_t), (uLongf *)&blen,

                  (Bytef *)p, size, level) != Z_OK) {

        error_report("Compress Failed!");

        return 0;

    }

    qemu_put_be32(f, blen);

    f->buf_index += blen;

    return blen + sizeof(int32_t);

}
