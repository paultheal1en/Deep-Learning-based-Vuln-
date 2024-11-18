static int ebml_read_ascii(AVIOContext *pb, int size, char **str)

{

    av_free(*str);

    /* EBML strings are usually not 0-terminated, so we allocate one

     * byte more, read the string and NULL-terminate it ourselves. */

    if (!(*str = av_malloc(size + 1)))

        return AVERROR(ENOMEM);

    if (avio_read(pb, (uint8_t *) *str, size) != size) {

        av_freep(str);

        return AVERROR(EIO);

    }

    (*str)[size] = '\0';



    return 0;

}
