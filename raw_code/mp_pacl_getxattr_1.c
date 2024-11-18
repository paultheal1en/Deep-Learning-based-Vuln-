static ssize_t mp_pacl_getxattr(FsContext *ctx, const char *path,

                                const char *name, void *value, size_t size)

{

    char *buffer;

    ssize_t ret;



    buffer = rpath(ctx, path);

    ret = lgetxattr(buffer, MAP_ACL_ACCESS, value, size);

    g_free(buffer);

    return ret;

}