int read_file(const char *filename, char **bufptr, size_t *size)

{

    FILE *f = fopen(filename, "rb");



    if (!f) {

        av_log(NULL, AV_LOG_ERROR, "Cannot read file '%s': %s\n", filename, strerror(errno));

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

    fread(*bufptr, 1, *size, f);

    (*bufptr)[*size++] = '\0';



    fclose(f);

    return 0;

}
