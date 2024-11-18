static int headroom(int *la)

{

    int l;

    if (*la == 0) {

        return 31;

    }

    l = 30 - av_log2(FFABS(*la));

    *la <<= l;

    return l;

}
