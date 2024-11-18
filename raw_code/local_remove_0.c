static int local_remove(FsContext *ctx, const char *path)

{

    int err;

    struct stat stbuf;

    char buffer[PATH_MAX];



    if (ctx->export_flags & V9FS_SM_MAPPED_FILE) {

        err =  lstat(rpath(ctx, path, buffer), &stbuf);

        if (err) {

            goto err_out;

        }

        /*

         * If directory remove .virtfs_metadata contained in the

         * directory

         */

        if (S_ISDIR(stbuf.st_mode)) {

            snprintf(buffer, ARRAY_SIZE(buffer), "%s/%s/%s",

                     ctx->fs_root, path, VIRTFS_META_DIR);

            err = remove(buffer);

            if (err < 0 && errno != ENOENT) {

                /*

                 * We didn't had the .virtfs_metadata file. May be file created

                 * in non-mapped mode ?. Ignore ENOENT.

                 */

                goto err_out;

            }

        }

        /*

         * Now remove the name from parent directory

         * .virtfs_metadata directory

         */

        err = remove(local_mapped_attr_path(ctx, path, buffer));

        if (err < 0 && errno != ENOENT) {

            /*

             * We didn't had the .virtfs_metadata file. May be file created

             * in non-mapped mode ?. Ignore ENOENT.

             */

            goto err_out;

        }

    }

    return remove(rpath(ctx, path, buffer));

err_out:

    return err;

}
