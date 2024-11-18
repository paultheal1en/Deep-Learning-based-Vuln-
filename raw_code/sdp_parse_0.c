static int sdp_parse(AVFormatContext *s, const char *content)

{

    const char *p;

    int letter;

    /* Some SDP lines, particularly for Realmedia or ASF RTSP streams,

     * contain long SDP lines containing complete ASF Headers (several

     * kB) or arrays of MDPR (RM stream descriptor) headers plus

     * "rulebooks" describing their properties. Therefore, the SDP line

     * buffer is large.

     *

     * The Vorbis FMTP line can be up to 16KB - see sdp_parse_fmtp. */

    char buf[16384], *q;

    SDPParseState sdp_parse_state, *s1 = &sdp_parse_state;



    memset(s1, 0, sizeof(SDPParseState));

    p = content;

    for(;;) {

        skip_spaces(&p);

        letter = *p;

        if (letter == '\0')

            break;

        p++;

        if (*p != '=')

            goto next_line;

        p++;

        /* get the content */

        q = buf;

        while (*p != '\n' && *p != '\r' && *p != '\0') {

            if ((q - buf) < sizeof(buf) - 1)

                *q++ = *p;

            p++;

        }

        *q = '\0';

        sdp_parse_line(s, s1, letter, buf);

    next_line:

        while (*p != '\n' && *p != '\0')

            p++;

        if (*p == '\n')

            p++;

    }

    return 0;

}
