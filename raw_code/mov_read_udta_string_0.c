static int mov_read_udta_string(MOVContext *c, ByteIOContext *pb, MOVAtom atom)

{

    char *str = NULL;

    int size;

    uint16_t str_size;



    if (c->itunes_metadata) {

        int data_size = get_be32(pb);

        int tag = get_le32(pb);

        if (tag == MKTAG('d','a','t','a')) {

            get_be32(pb); // type

            get_be32(pb); // unknown

            str_size = data_size - 16;

        } else return 0;

    } else {

        str_size = get_be16(pb); // string length

        get_be16(pb); // language

    }

    switch (atom.type) {

    case MKTAG(0xa9,'n','a','m'):

        str = c->fc->title; size = sizeof(c->fc->title); break;

    case MKTAG(0xa9,'A','R','T'):

    case MKTAG(0xa9,'w','r','t'):

        str = c->fc->author; size = sizeof(c->fc->author); break;

    case MKTAG(0xa9,'c','p','y'):

        str = c->fc->copyright; size = sizeof(c->fc->copyright); break;

    case MKTAG(0xa9,'c','m','t'):

    case MKTAG(0xa9,'i','n','f'):

        str = c->fc->comment; size = sizeof(c->fc->comment); break;

    case MKTAG(0xa9,'a','l','b'):

        str = c->fc->album; size = sizeof(c->fc->album); break;

    }

    if (!str)

        return 0;

    get_buffer(pb, str, FFMIN(size, str_size));

    dprintf(c->fc, "%.4s %s\n", (char*)&atom.type, str);

    return 0;

}
