int avio_get_str(AVIOContext *s, int maxlen, char *buf, int buflen)

{

    int i;



    // reserve 1 byte for terminating 0

    buflen = FFMIN(buflen - 1, maxlen);

    for (i = 0; i < buflen; i++)

        if (!(buf[i] = avio_r8(s)))

            return i + 1;

    if (buflen)

        buf[i] = 0;

    for (; i < maxlen; i++)

        if (!avio_r8(s))

            return i + 1;

    return maxlen;

}
