static int ftp_send_command(FTPContext *s, const char *command,
                            const int response_codes[], char **response)
{
    int err;
    ff_dlog(s, "%s", command);
    if (response)
        *response = NULL;
    if ((err = ffurl_write(s->conn_control, command, strlen(command))) < 0)
        return err;
    if (!err)
        return -1;
    /* return status */
    if (response_codes) {
        return ftp_status(s, response, response_codes);
    }
    return 0;
}