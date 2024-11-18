void qemu_iovec_concat_iov(QEMUIOVector *dst,
                           struct iovec *src_iov, unsigned int src_cnt,
                           size_t soffset, size_t sbytes)
{
    int i;
    size_t done;
    assert(dst->nalloc != -1);
    for (i = 0, done = 0; done < sbytes && i < src_cnt; i++) {
        if (soffset < src_iov[i].iov_len) {
            size_t len = MIN(src_iov[i].iov_len - soffset, sbytes - done);
            qemu_iovec_add(dst, src_iov[i].iov_base + soffset, len);
            done += len;
            soffset = 0;
        } else {
            soffset -= src_iov[i].iov_len;
    assert(soffset == 0); /* offset beyond end of src */