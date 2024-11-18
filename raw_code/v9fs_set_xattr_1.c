int v9fs_set_xattr(FsContext *ctx, const char *path, const char *name,

                   void *value, size_t size, int flags)

{

    XattrOperations *xops = get_xattr_operations(ctx->xops, name);

    if (xops) {

        return xops->setxattr(ctx, path, name, value, size, flags);

    }

    errno = -EOPNOTSUPP;

    return -1;



}
