static int mp_pacl_removexattr(FsContext *ctx,

                               const char *path, const char *name)

{

    int ret;

    char *buffer;



    buffer = rpath(ctx, path);

    ret  = lremovexattr(buffer, MAP_ACL_ACCESS);

    if (ret == -1 && errno == ENODATA) {

        /*

         * We don't get ENODATA error when trying to remove a

         * posix acl that is not present. So don't throw the error

         * even in case of mapped security model

         */

        errno = 0;

        ret = 0;

    }

    g_free(buffer);

    return ret;

}
