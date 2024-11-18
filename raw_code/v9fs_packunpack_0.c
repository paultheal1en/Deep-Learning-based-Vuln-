static size_t v9fs_packunpack(void *addr, struct iovec *sg, int sg_count,

                              size_t offset, size_t size, int pack)

{

    int i = 0;

    size_t copied = 0;



    for (i = 0; size && i < sg_count; i++) {

        size_t len;

        if (offset >= sg[i].iov_len) {

            /* skip this sg */

            offset -= sg[i].iov_len;

            continue;

        } else {

            len = MIN(sg[i].iov_len - offset, size);

            if (pack) {

                memcpy(sg[i].iov_base + offset, addr, len);

            } else {

                memcpy(addr, sg[i].iov_base + offset, len);

            }

            size -= len;

            copied += len;

            addr += len;

            if (size) {

                offset = 0;

                continue;

            }

        }

    }



    return copied;

}
