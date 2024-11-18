static int mp_user_setxattr(FsContext *ctx, const char *path, const char *name,

                            void *value, size_t size, int flags)

{

    char buffer[PATH_MAX];

    if (strncmp(name, "user.virtfs.", 12) == 0) {

        /*

         * Don't allow fetch of user.virtfs namesapce

         * in case of mapped security

         */

        errno = EACCES;

        return -1;

    }

    return lsetxattr(rpath(ctx, path, buffer), name, value, size, flags);

}
