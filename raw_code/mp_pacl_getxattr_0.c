static ssize_t mp_pacl_getxattr(FsContext *ctx, const char *path,

                                const char *name, void *value, size_t size)

{

    char buffer[PATH_MAX];

    return lgetxattr(rpath(ctx, path, buffer), MAP_ACL_ACCESS, value, size);

}
