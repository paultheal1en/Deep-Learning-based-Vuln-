static int ftp_status(FTPContext *s, char **line, const int response_codes[])

{

    int err, i, dash = 0, result = 0, code_found = 0;

    char buf[CONTROL_BUFFER_SIZE];

    AVBPrint line_buffer;



    if (line)

        av_bprint_init(&line_buffer, 0, AV_BPRINT_SIZE_AUTOMATIC);



    while (!code_found || dash) {

        if ((err = ftp_get_line(s, buf, sizeof(buf))) < 0) {

            av_bprint_finalize(&line_buffer, NULL);

            return err;

        }



        av_log(s, AV_LOG_DEBUG, "%s\n", buf);



        if (strlen(buf) < 4)

            continue;



        err = 0;

        for (i = 0; i < 3; ++i) {

            if (buf[i] < '0' || buf[i] > '9')

                continue;

            err *= 10;

            err += buf[i] - '0';

        }

        dash = !!(buf[3] == '-');



        for (i = 0; response_codes[i]; ++i) {

            if (err == response_codes[i]) {

                if (line)

                    av_bprintf(&line_buffer, "%s", buf);

                code_found = 1;

                result = err;

                break;

            }

        }

    }



    if (line)

        av_bprint_finalize(&line_buffer, line);

    return result;

}
