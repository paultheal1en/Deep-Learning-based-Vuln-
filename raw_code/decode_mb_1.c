static void decode_mb(MpegEncContext *s, int ref)

{

    s->dest[0] = s->current_picture.f.data[0] + (s->mb_y *  16                       * s->linesize)   + s->mb_x *  16;

    s->dest[1] = s->current_picture.f.data[1] + (s->mb_y * (16 >> s->chroma_y_shift) * s->uvlinesize) + s->mb_x * (16 >> s->chroma_x_shift);

    s->dest[2] = s->current_picture.f.data[2] + (s->mb_y * (16 >> s->chroma_y_shift) * s->uvlinesize) + s->mb_x * (16 >> s->chroma_x_shift);



    ff_init_block_index(s);

    ff_update_block_index(s);

    s->dest[1] += (16 >> s->chroma_x_shift) - 8;

    s->dest[2] += (16 >> s->chroma_x_shift) - 8;



    if (CONFIG_H264_DECODER && s->codec_id == AV_CODEC_ID_H264) {

        H264Context *h = (void*)s;

        h->mb_xy = s->mb_x + s->mb_y * s->mb_stride;

        memset(h->non_zero_count_cache, 0, sizeof(h->non_zero_count_cache));

        av_assert1(ref >= 0);

        /* FIXME: It is possible albeit uncommon that slice references

         * differ between slices. We take the easy approach and ignore

         * it for now. If this turns out to have any relevance in

         * practice then correct remapping should be added. */

        if (ref >= h->ref_count[0])

            ref = 0;

        if (!h->ref_list[0][ref].f.data[0]) {

            av_log(s->avctx, AV_LOG_DEBUG, "Reference not available for error concealing\n");

            ref = 0;






        fill_rectangle(&s->current_picture.f.ref_index[0][4 * h->mb_xy],

                       2, 2, 2, ref, 1);

        fill_rectangle(&h->ref_cache[0][scan8[0]], 4, 4, 8, ref, 1);

        fill_rectangle(h->mv_cache[0][scan8[0]], 4, 4, 8,

                       pack16to32(s->mv[0][0][0], s->mv[0][0][1]), 4);

        h->mb_mbaff =

        h->mb_field_decoding_flag = 0;

        ff_h264_hl_decode_mb(h);

    } else {

        assert(ref == 0);

        ff_MPV_decode_mb(s, s->block);

