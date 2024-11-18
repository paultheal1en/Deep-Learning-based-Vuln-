static int asf_write_header(AVFormatContext *s)

{

    ASFContext *asf = s->priv_data;



    asf->packet_size = PACKET_SIZE;

    asf->nb_packets = 0;



    asf->last_indexed_pts = 0;

    asf->index_ptr = (ASFIndex*)av_malloc( sizeof(ASFIndex) * ASF_INDEX_BLOCK );

    asf->nb_index_memory_alloc = ASF_INDEX_BLOCK;

    asf->nb_index_count = 0;

    asf->maximum_packet = 0;



    /* the data-chunk-size has to be 50, which is data_size - asf->data_offset

     *  at the moment this function is done. It is needed to use asf as

     *  streamable format. */

    if (asf_write_header1(s, 0, 50) < 0) {

        //av_free(asf);

        return -1;

    }



    put_flush_packet(s->pb);



    asf->packet_nb_payloads = 0;

    asf->packet_timestamp_start = -1;

    asf->packet_timestamp_end = -1;

    init_put_byte(&asf->pb, asf->packet_buf, asf->packet_size, 1,

                  NULL, NULL, NULL, NULL);



    return 0;

}
