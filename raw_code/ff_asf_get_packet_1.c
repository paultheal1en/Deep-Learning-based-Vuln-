static int ff_asf_get_packet(AVFormatContext *s, AVIOContext *pb)

{

    ASFContext *asf = s->priv_data;

    uint32_t packet_length, padsize;

    int rsize = 8;

    int c, d, e, off;



    // if we do not know packet size, allow skipping up to 32 kB

    off= 32768;

    if (s->packet_size > 0)

        off= (avio_tell(pb) - s->data_offset) % s->packet_size + 3;



    c=d=e=-1;

    while(off-- > 0){

        c=d; d=e;

        e= avio_r8(pb);

        if(c == 0x82 && !d && !e)

            break;




    if (c != 0x82) {

        /**

         * This code allows handling of -EAGAIN at packet boundaries (i.e.

         * if the packet sync code above triggers -EAGAIN). This does not

         * imply complete -EAGAIN handling support at random positions in

         * the stream.

         */

        if (pb->error == AVERROR(EAGAIN))

            return AVERROR(EAGAIN);

        if (!pb->eof_reached)

            av_log(s, AV_LOG_ERROR, "ff asf bad header %x  at:%"PRId64"\n", c, avio_tell(pb));


    if ((c & 0x8f) == 0x82) {

        if (d || e) {

            if (!pb->eof_reached)

                av_log(s, AV_LOG_ERROR, "ff asf bad non zero\n");



        c= avio_r8(pb);

        d= avio_r8(pb);

        rsize+=3;

    }else{

        avio_seek(pb, -1, SEEK_CUR); //FIXME




    asf->packet_flags    = c;

    asf->packet_property = d;



    DO_2BITS(asf->packet_flags >> 5, packet_length, s->packet_size);

    DO_2BITS(asf->packet_flags >> 1, padsize, 0); // sequence ignored

    DO_2BITS(asf->packet_flags >> 3, padsize, 0); // padding length



    //the following checks prevent overflows and infinite loops

    if(!packet_length || packet_length >= (1U<<29)){

        av_log(s, AV_LOG_ERROR, "invalid packet_length %d at:%"PRId64"\n", packet_length, avio_tell(pb));



    if(padsize >= packet_length){

        av_log(s, AV_LOG_ERROR, "invalid padsize %d at:%"PRId64"\n", padsize, avio_tell(pb));





    asf->packet_timestamp = avio_rl32(pb);

    avio_rl16(pb); /* duration */

    // rsize has at least 11 bytes which have to be present



    if (asf->packet_flags & 0x01) {

        asf->packet_segsizetype = avio_r8(pb); rsize++;

        asf->packet_segments = asf->packet_segsizetype & 0x3f;

    } else {

        asf->packet_segments = 1;

        asf->packet_segsizetype = 0x80;









    asf->packet_size_left = packet_length - padsize - rsize;

    if (packet_length < asf->hdr.min_pktsize)

        padsize += asf->hdr.min_pktsize - packet_length;

    asf->packet_padsize = padsize;

    av_dlog(s, "packet: size=%d padsize=%d  left=%d\n", s->packet_size, asf->packet_padsize, asf->packet_size_left);

    return 0;
