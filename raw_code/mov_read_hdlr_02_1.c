static int mov_read_hdlr(MOVContext *c, ByteIOContext *pb, MOVAtom atom)

{

    AVStream *st = c->fc->streams[c->fc->nb_streams-1];

    uint32_t type;

    uint32_t ctype;



    get_byte(pb); /* version */

    get_be24(pb); /* flags */



    /* component type */

    ctype = get_le32(pb);

    type = get_le32(pb); /* component subtype */



    dprintf(c->fc, "ctype= %c%c%c%c (0x%08x)\n", *((char *)&ctype), ((char *)&ctype)[1],

            ((char *)&ctype)[2], ((char *)&ctype)[3], (int) ctype);

    dprintf(c->fc, "stype= %c%c%c%c\n",

            *((char *)&type), ((char *)&type)[1], ((char *)&type)[2], ((char *)&type)[3]);

    if(!ctype)

        c->isom = 1;

    if     (type == MKTAG('v','i','d','e'))

        st->codec->codec_type = CODEC_TYPE_VIDEO;

    else if(type == MKTAG('s','o','u','n'))

        st->codec->codec_type = CODEC_TYPE_AUDIO;

    else if(type == MKTAG('m','1','a',' '))

        st->codec->codec_id = CODEC_ID_MP2;

    else if(type == MKTAG('s','u','b','p')) {

        st->codec->codec_type = CODEC_TYPE_SUBTITLE;

    }

    get_be32(pb); /* component  manufacture */

    get_be32(pb); /* component flags */

    get_be32(pb); /* component flags mask */



    if(atom.size <= 24)

        return 0; /* nothing left to read */



    url_fskip(pb, atom.size - (url_ftell(pb) - atom.offset));

    return 0;

}
