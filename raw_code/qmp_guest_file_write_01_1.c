GuestFileWrite *qmp_guest_file_write(int64_t handle, const char *buf_b64,
                                     bool has_count, int64_t count,
                                     Error **errp)
{
    GuestFileWrite *write_data = NULL;
    guchar *buf;
    gsize buf_len;
    int write_count;
    GuestFileHandle *gfh = guest_file_handle_find(handle, errp);
    FILE *fh;
    if (!gfh) {
    fh = gfh->fh;
    buf = g_base64_decode(buf_b64, &buf_len);
    if (!has_count) {
        count = buf_len;
    } else if (count < 0 || count > buf_len) {
        error_setg(errp, "value '%" PRId64 "' is invalid for argument count",
                   count);
        g_free(buf);
    write_count = fwrite(buf, 1, count, fh);
    if (ferror(fh)) {
        error_setg_errno(errp, errno, "failed to write to file");
        slog("guest-file-write failed, handle: %" PRId64, handle);
    } else {
        write_data = g_new0(GuestFileWrite, 1);
        write_data->count = write_count;
        write_data->eof = feof(fh);
        gfh->state = RW_STATE_WRITING;
    g_free(buf);
    clearerr(fh);
    return write_data;