int mov_write_ftyp_tag(ByteIOContext *pb, AVFormatContext *s)

{

    put_be32(pb, 0x14 ); /* size */

    put_tag(pb, "ftyp");



    if (!strcmp("3gp", s->oformat->name))

        put_tag(pb, "3gp4");

    else

        put_tag(pb, "isom");



    put_be32(pb, 0x200 );



    if (!strcmp("3gp", s->oformat->name))

        put_tag(pb, "3gp4");

    else

        put_tag(pb, "mp41");



    return 0x14;

}
