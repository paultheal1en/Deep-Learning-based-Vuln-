static ssize_t mp_user_getxattr(FsContext *ctx, const char *path,

                                const char *name, void *value, size_t size)

{

    char buffer[PATH_MAX];

    if (strncmp(name, "user.virtfs.", 12) == 0) {

        /*

         * Don't allow fetch of user.virtfs namesapce

         * in case of mapped security

         */

        errno = ENOATTR;

        return -1;

    }

    return lgetxattr(rpath(ctx, path, buffer), name, value, size);

}
