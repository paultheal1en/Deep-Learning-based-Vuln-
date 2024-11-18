size_t iov_to_buf(const struct iovec *iov, const unsigned int iov_cnt, size_t iov_off,

                  void *buf, size_t size)

{

    uint8_t *ptr;

    size_t iovec_off, buf_off;

    unsigned int i;



    ptr = buf;

    iovec_off = 0;

    buf_off = 0;

    for (i = 0; i < iov_cnt && size; i++) {

        if (iov_off < (iovec_off + iov[i].iov_len)) {

            size_t len = MIN((iovec_off + iov[i].iov_len) - iov_off , size);



            memcpy(ptr + buf_off, iov[i].iov_base + (iov_off - iovec_off), len);



            buf_off += len;

            iov_off += len;

            size -= len;

        }

        iovec_off += iov[i].iov_len;

    }

    return buf_off;

}
