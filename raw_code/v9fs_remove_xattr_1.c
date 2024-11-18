int v9fs_remove_xattr(FsContext *ctx,

                      const char *path, const char *name)

{

    XattrOperations *xops = get_xattr_operations(ctx->xops, name);

    if (xops) {

        return xops->removexattr(ctx, path, name);

    }

    errno = -EOPNOTSUPP;

    return -1;



}
