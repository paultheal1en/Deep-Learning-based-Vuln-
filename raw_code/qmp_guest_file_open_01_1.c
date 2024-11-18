int64_t qmp_guest_file_open(const char *path, bool has_mode, const char *mode,

                            Error **errp)

{

    FILE *fh;

    Error *local_err = NULL;

    int fd;

    int64_t ret = -1, handle;



    if (!has_mode) {

        mode = "r";

    }

    slog("guest-file-open called, filepath: %s, mode: %s", path, mode);

    fh = safe_open_or_create(path, mode, &local_err);

    if (local_err != NULL) {

        error_propagate(errp, local_err);

        return -1;

    }



    /* set fd non-blocking to avoid common use cases (like reading from a

     * named pipe) from hanging the agent

     */

    fd = fileno(fh);

    ret = fcntl(fd, F_GETFL);

    ret = fcntl(fd, F_SETFL, ret | O_NONBLOCK);

    if (ret == -1) {

        error_setg_errno(errp, errno, "failed to make file '%s' non-blocking",

                         path);

        fclose(fh);

        return -1;

    }



    handle = guest_file_handle_add(fh, errp);

    if (error_is_set(errp)) {

        fclose(fh);

        return -1;

    }



    slog("guest-file-open, handle: %" PRId64, handle);

    return handle;

}
