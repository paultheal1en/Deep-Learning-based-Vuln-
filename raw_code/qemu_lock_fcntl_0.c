static int qemu_lock_fcntl(int fd, int64_t start, int64_t len, int fl_type)

{

    int ret;

    struct flock fl = {

        .l_whence = SEEK_SET,

        .l_start  = start,

        .l_len    = len,

        .l_type   = fl_type,

    };

    ret = fcntl(fd, QEMU_SETLK, &fl);

    return ret == -1 ? -errno : 0;

}
