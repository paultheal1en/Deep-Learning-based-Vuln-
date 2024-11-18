static inline int qemu_gluster_zerofill(struct glfs_fd *fd, int64_t offset,

                                        int64_t size)

{

    return glfs_zerofill(fd, offset, size);

}
