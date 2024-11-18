enum CodecID codec_get_id(const CodecTag *tags, unsigned int tag)

{

    while (tags->id != 0) {

        if(   toupper((tag >> 0)&0xFF) == toupper((tags->tag >> 0)&0xFF)

           && toupper((tag >> 8)&0xFF) == toupper((tags->tag >> 8)&0xFF)

           && toupper((tag >>16)&0xFF) == toupper((tags->tag >>16)&0xFF)

           && toupper((tag >>24)&0xFF) == toupper((tags->tag >>24)&0xFF))

            return tags->id;

        tags++;

    }

    return CODEC_ID_NONE;

}
