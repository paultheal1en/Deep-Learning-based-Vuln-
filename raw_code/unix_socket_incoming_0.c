int unix_socket_incoming(const char *path)

{

    Error *local_err = NULL;

    int fd = unix_listen(path, NULL, 0, &local_err);



    if (local_err != NULL) {

        qerror_report_err(local_err);

        error_free(local_err);

    }

    return fd;

}
