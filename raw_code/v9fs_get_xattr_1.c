ssize_t v9fs_get_xattr(FsContext *ctx, const char *path,

                       const char *name, void *value, size_t size)

{

    XattrOperations *xops = get_xattr_operations(ctx->xops, name);

    if (xops) {

        return xops->getxattr(ctx, path, name, value, size);

    }

    errno = -EOPNOTSUPP;

    return -1;

}
