static int mp_dacl_setxattr(FsContext *ctx, const char *path, const char *name,

                            void *value, size_t size, int flags)

{

    char buffer[PATH_MAX];

    return lsetxattr(rpath(ctx, path, buffer), MAP_ACL_DEFAULT, value,

            size, flags);

}
