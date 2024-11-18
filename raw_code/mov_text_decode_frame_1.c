static int mov_text_decode_frame(AVCodecContext *avctx,

                            void *data, int *got_sub_ptr, AVPacket *avpkt)

{

    AVSubtitle *sub = data;

    int ret, ts_start, ts_end;

    AVBPrint buf;

    char *ptr = avpkt->data;

    char *end;

    //char *ptr_temp;

    int text_length, tsmb_type, style_entries, tsmb_size;

    int **style_start = {0,};

    int **style_end = {0,};

    int **style_flags = {0,};

    const uint8_t *tsmb;

    int index, i;

    int *flag;

    int *style_pos;



    if (!ptr || avpkt->size < 2)

        return AVERROR_INVALIDDATA;



    /*

     * A packet of size two with value zero is an empty subtitle

     * used to mark the end of the previous non-empty subtitle.

     * We can just drop them here as we have duration information

     * already. If the value is non-zero, then it's technically a

     * bad packet.

     */

    if (avpkt->size == 2)

        return AV_RB16(ptr) == 0 ? 0 : AVERROR_INVALIDDATA;



    /*

     * The first two bytes of the packet are the length of the text string

     * In complex cases, there are style descriptors appended to the string

     * so we can't just assume the packet size is the string size.

     */

    text_length = AV_RB16(ptr);

    end = ptr + FFMIN(2 + text_length, avpkt->size);

    ptr += 2;



    ts_start = av_rescale_q(avpkt->pts,

                            avctx->time_base,

                            (AVRational){1,100});

    ts_end   = av_rescale_q(avpkt->pts + avpkt->duration,

                            avctx->time_base,

                            (AVRational){1,100});



    tsmb_size = 0;

    // Note that the spec recommends lines be no longer than 2048 characters.

    av_bprint_init(&buf, 0, AV_BPRINT_SIZE_UNLIMITED);

    if (text_length + 2 != avpkt->size) {

        while (text_length + 2 + tsmb_size < avpkt->size)  {

            tsmb = ptr + text_length + tsmb_size;

            tsmb_size = AV_RB32(tsmb);

            tsmb += 4;

            tsmb_type = AV_RB32(tsmb);

            tsmb += 4;



            if (tsmb_type == MKBETAG('s','t','y','l')) {

                style_entries = AV_RB16(tsmb);

                tsmb += 2;



                for(i = 0; i < style_entries; i++) {

                    style_pos = av_malloc(4);

                    *style_pos = AV_RB16(tsmb);

                    index = i;

                    av_dynarray_add(&style_start, &index, style_pos);

                    tsmb += 2;

                    style_pos = av_malloc(4);

                    *style_pos = AV_RB16(tsmb);

                    index = i;

                    av_dynarray_add(&style_end, &index, style_pos);

                    tsmb += 2;

                    // fontID = AV_RB16(tsmb);

                    tsmb += 2;

                    flag = av_malloc(4);

                    *flag = AV_RB8(tsmb);

                    index = i;

                    av_dynarray_add(&style_flags, &index, flag);

                    //fontsize=AV_RB8(tsmb);

                    tsmb += 2;

                    // text-color-rgba

                    tsmb += 4;

                }

                text_to_ass(&buf, ptr, end, style_start, style_end, style_flags, style_entries);

                av_freep(&style_start);

                av_freep(&style_end);

                av_freep(&style_flags);

            }

        }

    } else

        text_to_ass(&buf, ptr, end, NULL, NULL, 0, 0);



    ret = ff_ass_add_rect_bprint(sub, &buf, ts_start, ts_end - ts_start);

    av_bprint_finalize(&buf, NULL);

    if (ret < 0)

        return ret;

    *got_sub_ptr = sub->num_rects > 0;

    return avpkt->size;

}
