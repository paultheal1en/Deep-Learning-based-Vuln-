static char *ctime1(char *buf2, int buf_size)

{

    time_t ti;

    char *p;



    ti = time(NULL);

    p = ctime(&ti);

    av_strlcpy(buf2, p, buf_size);

    p = buf2 + strlen(p) - 1;

    if (*p == '\n')

        *p = '\0';

    return buf2;

}
