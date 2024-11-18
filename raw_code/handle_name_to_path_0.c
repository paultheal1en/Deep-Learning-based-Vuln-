static int handle_name_to_path(FsContext *ctx, V9fsPath *dir_path,

                              const char *name, V9fsPath *target)

{

    char buffer[PATH_MAX];

    struct file_handle *fh;

    int dirfd, ret, mnt_id;

    struct handle_data *data = (struct handle_data *)ctx->private;



    /* "." and ".." are not allowed */

    if (!strcmp(name, ".") || !strcmp(name, "..")) {

        errno = EINVAL;

        return -1;



    }

    if (dir_path) {

        dirfd = open_by_handle(data->mountfd, dir_path->data, O_PATH);

    } else {

        /* relative to export root */

        dirfd = open(rpath(ctx, ".", buffer), O_DIRECTORY);

    }

    if (dirfd < 0) {

        return dirfd;

    }

    fh = g_malloc(sizeof(struct file_handle) + data->handle_bytes);

    fh->handle_bytes = data->handle_bytes;

    /* add a "./" at the beginning of the path */

    snprintf(buffer, PATH_MAX, "./%s", name);

    /* flag = 0 imply don't follow symlink */

    ret = name_to_handle(dirfd, buffer, fh, &mnt_id, 0);

    if (!ret) {

        target->data = (char *)fh;

        target->size = sizeof(struct file_handle) + data->handle_bytes;

    } else {

        g_free(fh);

    }

    close(dirfd);

    return ret;

}
