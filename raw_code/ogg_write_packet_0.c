static int ogg_write_packet(AVFormatContext *avfcontext,

			    int stream_index,

			    const uint8_t *buf, int size, int64_t pts)

{

    OggContext *context = avfcontext->priv_data ;

    AVCodecContext *avctx= &avfcontext->streams[stream_index]->codec;

    ogg_packet *op= &context->op;

    ogg_page og ;



    pts= av_rescale(pts, avctx->sample_rate, AV_TIME_BASE);



    if(!size){

//        av_log(avfcontext, AV_LOG_DEBUG, "zero packet\n");

        return 0;

    }

//    av_log(avfcontext, AV_LOG_DEBUG, "M%d\n", size);



    /* flush header packets so audio starts on a new page */



    if(!context->header_handled) {

	while(ogg_stream_flush(&context->os, &og)) {

	    put_buffer(&avfcontext->pb, og.header, og.header_len) ;

	    put_buffer(&avfcontext->pb, og.body, og.body_len) ;

	    put_flush_packet(&avfcontext->pb);

	}

	context->header_handled = 1 ;

    }



    op->packet = (uint8_t*) buf;

    op->bytes  = size;

    op->b_o_s  = op->packetno == 0;

    op->granulepos= pts;



    /* correct the fields in the packet -- essential for streaming */

                                                        

    ogg_stream_packetin(&context->os, op);              

                                                        

    while(ogg_stream_pageout(&context->os, &og)) {

        put_buffer(&avfcontext->pb, og.header, og.header_len);

	put_buffer(&avfcontext->pb, og.body, og.body_len);     

	put_flush_packet(&avfcontext->pb);

    }                                                   

    op->packetno++;



    return 0;

}
