int av_strerror(int errnum, char *errbuf, size_t errbuf_size)

{

    int ret = 0;

    const char *errstr = NULL;



    switch (errnum) {

    case AVERROR_EOF:               errstr = "End of file"; break;

    case AVERROR_INVALIDDATA:       errstr = "Invalid data found when processing input"; break;

    case AVERROR_NUMEXPECTED:       errstr = "Number syntax expected in filename"; break;

    case AVERROR_PATCHWELCOME:      errstr = "Not yet implemented in FFmpeg, patches welcome"; break;

    }



    if (errstr) {

        av_strlcpy(errbuf, errstr, errbuf_size);

    } else {

#if HAVE_STRERROR_R

        ret = strerror_r(AVUNERROR(errnum), errbuf, errbuf_size);

#endif

        if (!HAVE_STRERROR_R || ret < 0)

            snprintf(errbuf, errbuf_size, "Error number %d occurred", errnum);

    }



    return ret;

}
