static int ftp_store(FTPContext *s)

{

    char command[CONTROL_BUFFER_SIZE];

    const int stor_codes[] = {150, 0};



    snprintf(command, sizeof(command), "STOR %s\r\n", s->path);

    if (!ftp_send_command(s, command, stor_codes, NULL))

        return AVERROR(EIO);



    s->state = UPLOADING;



    return 0;

}
