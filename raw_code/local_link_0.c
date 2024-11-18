static int local_link(FsContext *ctx, V9fsPath *oldpath,

                      V9fsPath *dirpath, const char *name)

{

    int ret;

    V9fsString newpath;

    char buffer[PATH_MAX], buffer1[PATH_MAX];



    v9fs_string_init(&newpath);

    v9fs_string_sprintf(&newpath, "%s/%s", dirpath->data, name);



    ret = link(rpath(ctx, oldpath->data, buffer),

               rpath(ctx, newpath.data, buffer1));



    /* now link the virtfs_metadata files */

    if (!ret && (ctx->export_flags & V9FS_SM_MAPPED_FILE)) {

        /* Link the .virtfs_metadata files. Create the metada directory */

        ret = local_create_mapped_attr_dir(ctx, newpath.data);

        if (ret < 0) {

            goto err_out;

        }

        ret = link(local_mapped_attr_path(ctx, oldpath->data, buffer),

                   local_mapped_attr_path(ctx, newpath.data, buffer1));

        if (ret < 0 && errno != ENOENT) {

            goto err_out;

        }

    }

err_out:

    v9fs_string_free(&newpath);

    return ret;

}
