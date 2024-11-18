static int ftp_restart(FTPContext *s, int64_t pos)

{

    char command[CONTROL_BUFFER_SIZE];

    const int rest_codes[] = {350, 0};



    snprintf(command, sizeof(command), "REST %"PRId64"\r\n", pos);

    if (!ftp_send_command(s, command, rest_codes, NULL))

        return AVERROR(EIO);



    return 0;

}
