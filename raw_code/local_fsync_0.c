static int local_fsync(FsContext *ctx, int fd)

{

    if (0) /* Just to supress the warning. Will be removed in next patch. */

        (void)local_set_xattr(NULL, NULL);

    return fsync(fd);

}
