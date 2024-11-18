static int ftp_features(FTPContext *s)

{

    static const char *feat_command        = "FEAT\r\n";

    static const char *enable_utf8_command = "OPTS UTF8 ON\r\n";

    static const int feat_codes[] = {211, 0};

    static const int opts_codes[] = {200, 451};

    char *feat;



    if (ftp_send_command(s, feat_command, feat_codes, &feat) == 211) {

        if (av_stristr(feat, "UTF8"))

            ftp_send_command(s, enable_utf8_command, opts_codes, NULL);

    }

    return 0;

}
