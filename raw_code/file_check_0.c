static int file_check(URLContext *h, int mask)

{

#if HAVE_ACCESS && defined(R_OK)

    int ret = 0;

    if (access(h->filename, F_OK) < 0)

        return AVERROR(errno);

    if (mask&AVIO_FLAG_READ)

        if (access(h->filename, R_OK) >= 0)

            ret |= AVIO_FLAG_READ;

    if (mask&AVIO_FLAG_WRITE)

        if (access(h->filename, W_OK) >= 0)

            ret |= AVIO_FLAG_WRITE;

#else

    struct stat st;

    int ret = stat(h->filename, &st);

    if (ret < 0)

        return AVERROR(errno);



    ret |= st.st_mode&S_IRUSR ? mask&AVIO_FLAG_READ  : 0;

    ret |= st.st_mode&S_IWUSR ? mask&AVIO_FLAG_WRITE : 0;

#endif

    return ret;

}
