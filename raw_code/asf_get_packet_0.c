static int asf_get_packet(AVFormatContext *s)

{

    ASFContext *asf = s->priv_data;

    ByteIOContext *pb = &s->pb;

    uint32_t packet_length, padsize;

    int rsize = 9;

    int c;



    c = get_byte(pb);

    if (c != 0x82) {

        if (!url_feof(pb))

            av_log(s, AV_LOG_ERROR, "ff asf bad header %x  at:%"PRId64"\n", c, url_ftell(pb));

    }

    if ((c & 0x0f) == 2) { // always true for now

        if (get_le16(pb) != 0) {

            if (!url_feof(pb))

                av_log(s, AV_LOG_ERROR, "ff asf bad non zero\n");

            return AVERROR_IO;

        }

        rsize+=2;

/*    }else{

        if (!url_feof(pb))

            printf("ff asf bad header %x  at:%"PRId64"\n", c, url_ftell(pb));

        return AVERROR_IO;*/

    }



    asf->packet_flags = get_byte(pb);

    asf->packet_property = get_byte(pb);



    DO_2BITS(asf->packet_flags >> 5, packet_length, asf->packet_size);

    DO_2BITS(asf->packet_flags >> 1, padsize, 0); // sequence ignored

    DO_2BITS(asf->packet_flags >> 3, padsize, 0); // padding length



    //the following checks prevent overflows and infinite loops

    if(packet_length >= (1U<<29)){

        av_log(s, AV_LOG_ERROR, "invalid packet_length %d at:%"PRId64"\n", packet_length, url_ftell(pb));

        return -1;

    }

    if(padsize >= (1U<<29)){

        av_log(s, AV_LOG_ERROR, "invalid padsize %d at:%"PRId64"\n", padsize, url_ftell(pb));

        return -1;

    }



    asf->packet_timestamp = get_le32(pb);

    get_le16(pb); /* duration */

    // rsize has at least 11 bytes which have to be present



    if (asf->packet_flags & 0x01) {

        asf->packet_segsizetype = get_byte(pb); rsize++;

        asf->packet_segments = asf->packet_segsizetype & 0x3f;

    } else {

        asf->packet_segments = 1;

        asf->packet_segsizetype = 0x80;

    }

    asf->packet_size_left = packet_length - padsize - rsize;

    if (packet_length < asf->hdr.min_pktsize)

        padsize += asf->hdr.min_pktsize - packet_length;

    asf->packet_padsize = padsize;

#ifdef DEBUG

    printf("packet: size=%d padsize=%d  left=%d\n", asf->packet_size, asf->packet_padsize, asf->packet_size_left);

#endif

    return 0;

}
