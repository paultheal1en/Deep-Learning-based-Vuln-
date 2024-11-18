static int swf_write_packet(AVFormatContext *s, int stream_index, 

                           const uint8_t *buf, int size, int64_t pts)

{

    AVCodecContext *codec = &s->streams[stream_index]->codec;

    if (codec->codec_type == CODEC_TYPE_AUDIO)

        return swf_write_audio(s, buf, size);

    else

        return swf_write_video(s, codec, buf, size);

}
