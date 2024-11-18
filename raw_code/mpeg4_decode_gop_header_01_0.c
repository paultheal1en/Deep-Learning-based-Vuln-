static int mpeg4_decode_gop_header(MpegEncContext * s, GetBitContext *gb){

    int hours, minutes, seconds;



    if(!show_bits(gb, 18)){

        av_log(s->avctx, AV_LOG_WARNING, "GOP header invalid\n");

        return -1;

    }



    hours= get_bits(gb, 5);

    minutes= get_bits(gb, 6);

    skip_bits1(gb);

    seconds= get_bits(gb, 6);



    s->time_base= seconds + 60*(minutes + 60*hours);



    skip_bits1(gb);

    skip_bits1(gb);



    return 0;

}
