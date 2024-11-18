static int rtp_parse_mp4_au(PayloadContext *data, const uint8_t *buf)

{

    int au_headers_length, au_header_size, i;

    GetBitContext getbitcontext;



    /* decode the first 2 bytes where the AUHeader sections are stored

       length in bits */

    au_headers_length = AV_RB16(buf);



    if (au_headers_length > RTP_MAX_PACKET_LENGTH)

      return -1;



    data->au_headers_length_bytes = (au_headers_length + 7) / 8;



    /* skip AU headers length section (2 bytes) */

    buf += 2;



    init_get_bits(&getbitcontext, buf, data->au_headers_length_bytes * 8);



    /* XXX: Wrong if optionnal additional sections are present (cts, dts etc...) */

    au_header_size = data->sizelength + data->indexlength;

    if (au_header_size <= 0 || (au_headers_length % au_header_size != 0))

        return -1;



    data->nb_au_headers = au_headers_length / au_header_size;

    if (!data->au_headers || data->au_headers_allocated < data->nb_au_headers) {

        av_free(data->au_headers);

        data->au_headers = av_malloc(sizeof(struct AUHeaders) * data->nb_au_headers);

        if (!data->au_headers)

            return AVERROR(ENOMEM);

        data->au_headers_allocated = data->nb_au_headers;

    }



    /* XXX: We handle multiple AU Section as only one (need to fix this for interleaving)

       In my test, the FAAD decoder does not behave correctly when sending each AU one by one

       but does when sending the whole as one big packet...  */

    data->au_headers[0].size = 0;

    data->au_headers[0].index = 0;

    for (i = 0; i < data->nb_au_headers; ++i) {

        data->au_headers[0].size += get_bits_long(&getbitcontext, data->sizelength);

        data->au_headers[0].index = get_bits_long(&getbitcontext, data->indexlength);

    }



    data->nb_au_headers = 1;



    return 0;

}
