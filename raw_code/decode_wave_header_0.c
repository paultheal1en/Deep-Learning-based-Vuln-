static int decode_wave_header(AVCodecContext *avctx, const uint8_t *header,

                              int header_size)

{

    int len;

    short wave_format;





    if (bytestream_get_le32(&header) != MKTAG('R','I','F','F')) {

        av_log(avctx, AV_LOG_ERROR, "missing RIFF tag\n");

        return -1;

    }



    header += 4; /* chunk size */;



    if (bytestream_get_le32(&header) != MKTAG('W','A','V','E')) {

        av_log(avctx, AV_LOG_ERROR, "missing WAVE tag\n");

        return -1;

    }



    while (bytestream_get_le32(&header) != MKTAG('f','m','t',' ')) {

        len = bytestream_get_le32(&header);

        header += len;

    }

    len = bytestream_get_le32(&header);



    if (len < 16) {

        av_log(avctx, AV_LOG_ERROR, "fmt chunk was too short\n");

        return -1;

    }



    wave_format = bytestream_get_le16(&header);



    switch (wave_format) {

        case WAVE_FORMAT_PCM:

            break;

        default:

            av_log(avctx, AV_LOG_ERROR, "unsupported wave format\n");

            return -1;

    }



    header += 2;        // skip channels    (already got from shorten header)

    avctx->sample_rate = bytestream_get_le32(&header);

    header += 4;        // skip bit rate    (represents original uncompressed bit rate)

    header += 2;        // skip block align (not needed)

    avctx->bits_per_coded_sample = bytestream_get_le16(&header);



    if (avctx->bits_per_coded_sample != 16) {

        av_log(avctx, AV_LOG_ERROR, "unsupported number of bits per sample\n");

        return -1;

    }



    len -= 16;

    if (len > 0)

        av_log(avctx, AV_LOG_INFO, "%d header bytes unparsed\n", len);



    return 0;

}
