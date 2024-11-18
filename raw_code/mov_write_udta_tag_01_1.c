static int mov_write_udta_tag(ByteIOContext *pb, MOVContext* mov,

                              AVFormatContext *s)

{

    int pos = url_ftell(pb);

    int i;



    put_be32(pb, 0); /* size */

    put_tag(pb, "udta");



    /* iTunes meta data */

    mov_write_meta_tag(pb, mov, s);



    /* Requirements */

    for (i=0; i<MAX_STREAMS; i++) {

        if(mov->tracks[i].entry <= 0) continue;

        if (mov->tracks[i].enc->codec_id == CODEC_ID_AAC ||

            mov->tracks[i].enc->codec_id == CODEC_ID_MPEG4) {

            int pos = url_ftell(pb);

            put_be32(pb, 0); /* size */

            put_tag(pb, "\251req");

            put_be16(pb, sizeof("QuickTime 6.0 or greater") - 1);

            put_be16(pb, 0);

            put_buffer(pb, "QuickTime 6.0 or greater",

                       sizeof("QuickTime 6.0 or greater") - 1);

            updateSize(pb, pos);

            break;

        }

    }



    /* Encoder */

    if(!(mov->tracks[0].enc->flags & CODEC_FLAG_BITEXACT))

    {

        int pos = url_ftell(pb);

        put_be32(pb, 0); /* size */

        put_tag(pb, "\251enc");

        put_be16(pb, sizeof(LIBAVFORMAT_IDENT) - 1); /* string length */

        put_be16(pb, 0);

        put_buffer(pb, LIBAVFORMAT_IDENT, sizeof(LIBAVFORMAT_IDENT) - 1);

        updateSize(pb, pos);

    }



    if( s->title[0] )

    {

        int pos = url_ftell(pb);

        put_be32(pb, 0); /* size */

        put_tag(pb, "\251nam");

        put_be16(pb, strlen(s->title)); /* string length */

        put_be16(pb, 0);

        put_buffer(pb, s->title, strlen(s->title));

        updateSize(pb, pos);

    }



    if( s->author[0] )

    {

        int pos = url_ftell(pb);

        put_be32(pb, 0); /* size */

        put_tag(pb, /*"\251aut"*/ "\251day" );

        put_be16(pb, strlen(s->author)); /* string length */

        put_be16(pb, 0);

        put_buffer(pb, s->author, strlen(s->author));

        updateSize(pb, pos);

    }



    if( s->comment[0] )

    {

        int pos = url_ftell(pb);

        put_be32(pb, 0); /* size */

        put_tag(pb, "\251des");

        put_be16(pb, strlen(s->comment)); /* string length */

        put_be16(pb, 0);

        put_buffer(pb, s->comment, strlen(s->comment));

        updateSize(pb, pos);

    }



    return updateSize(pb, pos);

}
