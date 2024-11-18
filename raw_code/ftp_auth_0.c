static int ftp_auth(FTPContext *s, char *auth)

{

    const char *user = NULL, *pass = NULL;

    char *end = NULL, buf[CONTROL_BUFFER_SIZE];

    int err;

    av_assert2(auth);



    user = av_strtok(auth, ":", &end);

    pass = av_strtok(end, ":", &end);



    if (user) {

        snprintf(buf, sizeof(buf), "USER %s\r\n", user);

        if ((err = ffurl_write(s->conn_control, buf, strlen(buf))) < 0)

            return err;

        ftp_status(s, &err, NULL, NULL, NULL, -1);

        if (err == 3) {

            if (pass) {

                snprintf(buf, sizeof(buf), "PASS %s\r\n", pass);

                if ((err = ffurl_write(s->conn_control, buf, strlen(buf))) < 0)

                    return err;

                ftp_status(s, &err, NULL, NULL, NULL, -1);

            } else

                return AVERROR(EACCES);

        }

        if (err != 2) {

            return AVERROR(EACCES);

        }

    } else {

        const char* command = "USER anonymous\r\n";

        if ((err = ffurl_write(s->conn_control, command, strlen(command))) < 0)

            return err;

        ftp_status(s, &err, NULL, NULL, NULL, -1);

        if (err == 3) {

            if (s->anonymous_password) {

                snprintf(buf, sizeof(buf), "PASS %s\r\n", s->anonymous_password);

            } else

                snprintf(buf, sizeof(buf), "PASS nopassword\r\n");

            if ((err = ffurl_write(s->conn_control, buf, strlen(buf))) < 0)

                return err;

            ftp_status(s, &err, NULL, NULL, NULL, -1);

        }

        if (err != 2) {

            return AVERROR(EACCES);

        }

    }



    return 0;

}
