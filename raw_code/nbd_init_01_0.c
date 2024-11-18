int nbd_init(int fd, QIOChannelSocket *sioc, NBDExportInfo *info,

             Error **errp)

{

    unsigned long sectors = info->size / BDRV_SECTOR_SIZE;

    if (info->size / BDRV_SECTOR_SIZE != sectors) {

        error_setg(errp, "Export size %" PRIu64 " too large for 32-bit kernel",

                   info->size);

        return -E2BIG;

    }



    trace_nbd_init_set_socket();



    if (ioctl(fd, NBD_SET_SOCK, (unsigned long) sioc->fd) < 0) {

        int serrno = errno;

        error_setg(errp, "Failed to set NBD socket");

        return -serrno;

    }



    trace_nbd_init_set_block_size(BDRV_SECTOR_SIZE);



    if (ioctl(fd, NBD_SET_BLKSIZE, (unsigned long)BDRV_SECTOR_SIZE) < 0) {

        int serrno = errno;

        error_setg(errp, "Failed setting NBD block size");

        return -serrno;

    }



    trace_nbd_init_set_size(sectors);

    if (info->size % BDRV_SECTOR_SIZE) {

        trace_nbd_init_trailing_bytes(info->size % BDRV_SECTOR_SIZE);

    }



    if (ioctl(fd, NBD_SET_SIZE_BLOCKS, sectors) < 0) {

        int serrno = errno;

        error_setg(errp, "Failed setting size (in blocks)");

        return -serrno;

    }



    if (ioctl(fd, NBD_SET_FLAGS, (unsigned long) info->flags) < 0) {

        if (errno == ENOTTY) {

            int read_only = (info->flags & NBD_FLAG_READ_ONLY) != 0;

            trace_nbd_init_set_readonly();



            if (ioctl(fd, BLKROSET, (unsigned long) &read_only) < 0) {

                int serrno = errno;

                error_setg(errp, "Failed setting read-only attribute");

                return -serrno;

            }

        } else {

            int serrno = errno;

            error_setg(errp, "Failed setting flags");

            return -serrno;

        }

    }



    trace_nbd_init_finish();



    return 0;

}
