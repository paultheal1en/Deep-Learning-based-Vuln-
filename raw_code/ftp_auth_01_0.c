static int ftp_auth(FTPContext *s)

{

    const char *user = NULL, *pass = NULL;

    char *end = NULL, buf[CONTROL_BUFFER_SIZE], credencials[CREDENTIALS_BUFFER_SIZE];

    int err;

    const int user_codes[] = {331, 230, 0};

    const int pass_codes[] = {230, 0};



    /* Authentication may be repeated, original string has to be saved */

    av_strlcpy(credencials, s->credencials, sizeof(credencials));



    user = av_strtok(credencials, ":", &end);

    pass = av_strtok(end, ":", &end);



    if (!user) {

        user = "anonymous";

        pass = s->anonymous_password ? s->anonymous_password : "nopassword";

    }



    snprintf(buf, sizeof(buf), "USER %s\r\n", user);

    err = ftp_send_command(s, buf, user_codes, NULL);

    if (err == 331) {

        if (pass) {

            snprintf(buf, sizeof(buf), "PASS %s\r\n", pass);

            err = ftp_send_command(s, buf, pass_codes, NULL);

        } else

            return AVERROR(EACCES);

    }

    if (!err)

        return AVERROR(EACCES);



    return 0;

}
