static int ftp_retrieve(FTPContext *s)

{

    char command[CONTROL_BUFFER_SIZE];

    const int retr_codes[] = {150, 550, 0}; /* 550 is incorrect code */



    snprintf(command, sizeof(command), "RETR %s\r\n", s->path);

    if (ftp_send_command(s, command, retr_codes, NULL) != 150)

        return AVERROR(EIO);



    s->state = DOWNLOADING;



    return 0;

}
