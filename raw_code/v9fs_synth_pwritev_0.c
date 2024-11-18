static ssize_t v9fs_synth_pwritev(FsContext *ctx, V9fsFidOpenState *fs,

                                  const struct iovec *iov,

                                  int iovcnt, off_t offset)

{

    int i, count = 0, wcount;

    V9fsSynthOpenState *synth_open = fs->private;

    V9fsSynthNode *node = synth_open->node;

    if (!node->attr->write) {

        errno = EPERM;

        return -1;

    }

    for (i = 0; i < iovcnt; i++) {

        wcount = node->attr->write(iov[i].iov_base, iov[i].iov_len,

                                   offset, node->private);

        offset += wcount;

        count  += wcount;

        /* If we wrote less than requested. we are done */

        if (wcount < iov[i].iov_len) {

            break;

        }

    }

    return count;

}
