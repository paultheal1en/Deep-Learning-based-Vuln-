static int query_codec(enum CodecID id, int std_compliance)

{

    CodecMime *cm= ff_id3v2_mime_tags;

    while(cm->id != CODEC_ID_NONE) {

        if(id == cm->id)

            return MKTAG('A', 'P', 'I', 'C');

        cm++;

    }

    return -1;

}
