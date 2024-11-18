static ssize_t proxy_pwritev(FsContext *ctx, V9fsFidOpenState *fs,

                             const struct iovec *iov,

                             int iovcnt, off_t offset)

{

    ssize_t ret;



#ifdef CONFIG_PREADV

    ret = pwritev(fs->fd, iov, iovcnt, offset);

#else

    ret = lseek(fs->fd, offset, SEEK_SET);

    if (ret >= 0) {

        ret = writev(fs->fd, iov, iovcnt);

    }

#endif

#ifdef CONFIG_SYNC_FILE_RANGE

    if (ret > 0 && ctx->export_flags & V9FS_IMMEDIATE_WRITEOUT) {

        /*

         * Initiate a writeback. This is not a data integrity sync.

         * We want to ensure that we don't leave dirty pages in the cache

         * after write when writeout=immediate is sepcified.

         */

        sync_file_range(fs->fd, offset, ret,

                        SYNC_FILE_RANGE_WAIT_BEFORE | SYNC_FILE_RANGE_WRITE);

    }

#endif

    return ret;

}
