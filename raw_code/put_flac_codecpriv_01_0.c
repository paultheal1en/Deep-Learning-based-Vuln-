static int put_flac_codecpriv(AVFormatContext *s, ByteIOContext *pb, AVCodecContext *codec)

{

    // if the extradata_size is greater than FLAC_STREAMINFO_SIZE,

    // assume that it's in Matroska format already

    if (codec->extradata_size < FLAC_STREAMINFO_SIZE) {

        av_log(s, AV_LOG_ERROR, "Invalid FLAC extradata\n");

        return -1;

    } else if (codec->extradata_size == FLAC_STREAMINFO_SIZE) {

        // only the streaminfo packet

        put_buffer(pb, "fLaC", 4);

        put_byte(pb, 0x80);

        put_be24(pb, FLAC_STREAMINFO_SIZE);

    } else if(memcmp("fLaC", codec->extradata, 4)) {

        av_log(s, AV_LOG_ERROR, "Invalid FLAC extradata\n");

        return -1;

    }

    put_buffer(pb, codec->extradata, codec->extradata_size);

    return 0;

}
