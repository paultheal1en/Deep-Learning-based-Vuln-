int cmdutils_read_file(const char *filename, char **bufptr, size_t *size)

{

    int ret;

    FILE *f = fopen(filename, "rb");



    if (!f) {

        av_log(NULL, AV_LOG_ERROR, "Cannot read file '%s': %s\n", filename,

               strerror(errno));

        return AVERROR(errno);

    }

    fseek(f, 0, SEEK_END);

    *size = ftell(f);

    fseek(f, 0, SEEK_SET);

    *bufptr = av_malloc(*size + 1);

    if (!*bufptr) {

        av_log(NULL, AV_LOG_ERROR, "Could not allocate file buffer\n");

        fclose(f);

        return AVERROR(ENOMEM);

    }

    ret = fread(*bufptr, 1, *size, f);

    if (ret < *size) {

        av_free(*bufptr);

        if (ferror(f)) {

            av_log(NULL, AV_LOG_ERROR, "Error while reading file '%s': %s\n",

                   filename, strerror(errno));

            ret = AVERROR(errno);

        } else

            ret = AVERROR_EOF;

    } else {

        ret = 0;

        (*bufptr)[(*size)++] = '\0';

    }



    fclose(f);

    return ret;

}
