static void rtsp_parse_rtp_info(RTSPState *rt, const char *p)

{

    int read = 0;

    char key[20], value[1024], url[1024] = "";

    uint32_t seq = 0, rtptime = 0;



    for (;;) {

        p += strspn(p, SPACE_CHARS);

        if (!*p)

            break;

        get_word_sep(key, sizeof(key), "=", &p);

        if (*p != '=')

            break;

        p++;

        get_word_sep(value, sizeof(value), ";, ", &p);

        read++;

        if (!strcmp(key, "url"))

            av_strlcpy(url, value, sizeof(url));

        else if (!strcmp(key, "seq"))

            seq = strtol(value, NULL, 10);

        else if (!strcmp(key, "rtptime"))

            rtptime = strtol(value, NULL, 10);

        if (*p == ',') {

            handle_rtp_info(rt, url, seq, rtptime);

            url[0] = '\0';

            seq = rtptime = 0;

            read = 0;

        }

        if (*p)

            p++;

    }

    if (read > 0)

        handle_rtp_info(rt, url, seq, rtptime);

}
