static int check_tag(AVIOContext *s, int offset, unsigned int len)

{

    char tag[4];



    if (len > 4 ||

        avio_seek(s, offset, SEEK_SET) < 0 ||

        avio_read(s, tag, len) < len)

        return -1;

    else if (!AV_RB32(tag) || is_tag(tag, len))

        return 1;



    return 0;

}
