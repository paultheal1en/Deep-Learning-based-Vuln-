int nbd_init(int fd, int csock, uint32_t flags, off_t size, size_t blocksize)

{

    TRACE("Setting NBD socket");



    if (ioctl(fd, NBD_SET_SOCK, csock) < 0) {

        int serrno = errno;

        LOG("Failed to set NBD socket");

        errno = serrno;

        return -1;

    }



    TRACE("Setting block size to %lu", (unsigned long)blocksize);



    if (ioctl(fd, NBD_SET_BLKSIZE, blocksize) < 0) {

        int serrno = errno;

        LOG("Failed setting NBD block size");

        errno = serrno;

        return -1;

    }



        TRACE("Setting size to %zd block(s)", (size_t)(size / blocksize));



    if (ioctl(fd, NBD_SET_SIZE_BLOCKS, size / blocksize) < 0) {

        int serrno = errno;

        LOG("Failed setting size (in blocks)");

        errno = serrno;

        return -1;

    }



    if (flags & NBD_FLAG_READ_ONLY) {

        int read_only = 1;

        TRACE("Setting readonly attribute");



        if (ioctl(fd, BLKROSET, (unsigned long) &read_only) < 0) {

            int serrno = errno;

            LOG("Failed setting read-only attribute");

            errno = serrno;

            return -1;

        }

    }



    if (ioctl(fd, NBD_SET_FLAGS, flags) < 0

        && errno != ENOTTY) {

        int serrno = errno;

        LOG("Failed setting flags");

        errno = serrno;

        return -1;

    }



    TRACE("Negotiation ended");



    return 0;

}
