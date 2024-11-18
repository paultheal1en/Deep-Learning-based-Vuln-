static int ftp_current_dir(FTPContext *s)

{

    char *res = NULL, *start = NULL, *end = NULL;

    int i;

    const char *command = "PWD\r\n";

    const int pwd_codes[] = {257, 0};



    if (!ftp_send_command(s, command, pwd_codes, &res))

        goto fail;



    for (i = 0; res[i]; ++i) {

        if (res[i] == '"') {

            if (!start) {

                start = res + i + 1;

                continue;

            }

            end = res + i;

            break;

        }

    }



    if (!end)

        goto fail;



    if (end > res && end[-1] == '/') {

        end[-1] = '\0';

    } else

        *end = '\0';

    av_strlcpy(s->path, start, sizeof(s->path));



    av_free(res);

    return 0;



  fail:

    av_free(res);

    return AVERROR(EIO);

}
