static int mkv_write_native_codecprivate(AVFormatContext *s,

                                         AVCodecParameters *par,

                                         AVIOContext *dyn_cp)

{

    switch (par->codec_id) {

    case AV_CODEC_ID_VORBIS:

    case AV_CODEC_ID_THEORA:

        return put_xiph_codecpriv(s, dyn_cp, par);

    case AV_CODEC_ID_FLAC:

        return put_flac_codecpriv(s, dyn_cp, par);

    case AV_CODEC_ID_WAVPACK:

        return put_wv_codecpriv(dyn_cp, par);

    case AV_CODEC_ID_H264:

        return ff_isom_write_avcc(dyn_cp, par->extradata,

                                  par->extradata_size);

    case AV_CODEC_ID_HEVC:

        ff_isom_write_hvcc(dyn_cp, par->extradata,

                           par->extradata_size, 0);

        return 0;

    case AV_CODEC_ID_ALAC:

        if (par->extradata_size < 36) {

            av_log(s, AV_LOG_ERROR,

                   "Invalid extradata found, ALAC expects a 36-byte "

                   "QuickTime atom.");

            return AVERROR_INVALIDDATA;

        } else

            avio_write(dyn_cp, par->extradata + 12,

                       par->extradata_size - 12);

        break;

    default:

        if (par->codec_id == AV_CODEC_ID_PRORES &&

            ff_codec_get_id(ff_codec_movvideo_tags, par->codec_tag) == AV_CODEC_ID_PRORES) {

            avio_wl32(dyn_cp, par->codec_tag);

        } else if (par->extradata_size && par->codec_id != AV_CODEC_ID_TTA)

            avio_write(dyn_cp, par->extradata, par->extradata_size);

    }



    return 0;

}
