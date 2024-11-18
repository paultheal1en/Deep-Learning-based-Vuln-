net_checksum_add_iov(const struct iovec *iov, const unsigned int iov_cnt,

                     uint32_t iov_off, uint32_t size)

{

    size_t iovec_off, buf_off;

    unsigned int i;

    uint32_t res = 0;

    uint32_t seq = 0;



    iovec_off = 0;

    buf_off = 0;

    for (i = 0; i < iov_cnt && size; i++) {

        if (iov_off < (iovec_off + iov[i].iov_len)) {

            size_t len = MIN((iovec_off + iov[i].iov_len) - iov_off , size);

            void *chunk_buf = iov[i].iov_base + (iov_off - iovec_off);



            res += net_checksum_add_cont(len, chunk_buf, seq);

            seq += len;



            buf_off += len;

            iov_off += len;

            size -= len;

        }

        iovec_off += iov[i].iov_len;

    }

    return res;

}
