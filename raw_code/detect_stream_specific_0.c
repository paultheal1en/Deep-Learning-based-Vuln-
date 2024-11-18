static int detect_stream_specific(AVFormatContext *avf, int idx)

{

    ConcatContext *cat = avf->priv_data;

    AVStream *st = cat->avf->streams[idx];

    ConcatStream *cs = &cat->cur_file->streams[idx];

    AVBitStreamFilterContext *bsf;

    int ret;



    if (cat->auto_convert && st->codecpar->codec_id == AV_CODEC_ID_H264 &&

        (st->codecpar->extradata_size < 4 || AV_RB32(st->codecpar->extradata) != 1)) {

        av_log(cat->avf, AV_LOG_INFO,

               "Auto-inserting h264_mp4toannexb bitstream filter\n");

        if (!(bsf = av_bitstream_filter_init("h264_mp4toannexb"))) {

            av_log(avf, AV_LOG_ERROR, "h264_mp4toannexb bitstream filter "

                   "required for H.264 streams\n");

            return AVERROR_BSF_NOT_FOUND;

        }

        cs->bsf = bsf;



        cs->avctx = avcodec_alloc_context3(NULL);

        if (!cs->avctx)

            return AVERROR(ENOMEM);



        /* This really should be part of the bsf work.

           Note: input bitstream filtering will not work with bsf that

           create extradata from the first packet. */

        av_freep(&st->codecpar->extradata);

        st->codecpar->extradata_size = 0;



        ret = avcodec_parameters_to_context(cs->avctx, st->codecpar);

        if (ret < 0) {

            avcodec_free_context(&cs->avctx);

            return ret;

        }



    }

    return 0;

}
