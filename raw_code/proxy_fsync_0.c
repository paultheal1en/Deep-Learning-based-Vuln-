static int proxy_fsync(FsContext *ctx, int fid_type,

                       V9fsFidOpenState *fs, int datasync)

{

    int fd;



    if (fid_type == P9_FID_DIR) {

        fd = dirfd(fs->dir);

    } else {

        fd = fs->fd;

    }



    if (datasync) {

        return qemu_fdatasync(fd);

    } else {

        return fsync(fd);

    }

}
