static int mp3_write_xing(AVFormatContext *s)

{

    MP3Context       *mp3 = s->priv_data;

    AVCodecContext *codec = s->streams[mp3->audio_stream_idx]->codec;

    AVDictionaryEntry *enc = av_dict_get(s->streams[mp3->audio_stream_idx]->metadata, "encoder", NULL, 0);

    AVIOContext *dyn_ctx;

    int32_t        header;

    MPADecodeHeader  mpah;

    int srate_idx, i, channels;

    int bitrate_idx;

    int best_bitrate_idx = -1;

    int best_bitrate_error = INT_MAX;

    int ret;

    int ver = 0;

    int bytes_needed;



    if (!s->pb->seekable || !mp3->write_xing)

        return 0;



    for (i = 0; i < FF_ARRAY_ELEMS(avpriv_mpa_freq_tab); i++) {

        const uint16_t base_freq = avpriv_mpa_freq_tab[i];



        if      (codec->sample_rate == base_freq)     ver = 0x3; // MPEG 1

        else if (codec->sample_rate == base_freq / 2) ver = 0x2; // MPEG 2

        else if (codec->sample_rate == base_freq / 4) ver = 0x0; // MPEG 2.5

        else continue;



        srate_idx = i;

        break;

    }

    if (i == FF_ARRAY_ELEMS(avpriv_mpa_freq_tab)) {

        av_log(s, AV_LOG_WARNING, "Unsupported sample rate, not writing Xing header.\n");

        return -1;

    }



    switch (codec->channels) {

    case 1:  channels = MPA_MONO;                                          break;

    case 2:  channels = MPA_STEREO;                                        break;

    default: av_log(s, AV_LOG_WARNING, "Unsupported number of channels, "

                    "not writing Xing header.\n");

             return -1;

    }



    /* dummy MPEG audio header */

    header  =  0xffU                                 << 24; // sync

    header |= (0x7 << 5 | ver << 3 | 0x1 << 1 | 0x1) << 16; // sync/audio-version/layer 3/no crc*/

    header |= (srate_idx << 2) << 8;

    header |= channels << 6;



    for (bitrate_idx = 1; bitrate_idx < 15; bitrate_idx++) {

        int bit_rate = 1000 * avpriv_mpa_bitrate_tab[ver != 3][3 - 1][bitrate_idx];

        int error    = FFABS(bit_rate - codec->bit_rate);



        if (error < best_bitrate_error) {

            best_bitrate_error = error;

            best_bitrate_idx   = bitrate_idx;

        }

    }

    av_assert0(best_bitrate_idx >= 0);



    for (bitrate_idx = best_bitrate_idx; ; bitrate_idx++) {

        int32_t mask = bitrate_idx << (4 + 8);

        if (15 == bitrate_idx)

            return -1;

        header |= mask;



        avpriv_mpegaudio_decode_header(&mpah, header);

        mp3->xing_offset = xing_offtbl[mpah.lsf == 1][mpah.nb_channels == 1] + 4;

        bytes_needed     = mp3->xing_offset + XING_SIZE;



        if (bytes_needed <= mpah.frame_size)

            break;



        header &= ~mask;

    }



    ret = avio_open_dyn_buf(&dyn_ctx);

    if (ret < 0)

        return ret;



    avio_wb32(dyn_ctx, header);



    ffio_fill(dyn_ctx, 0, mp3->xing_offset - 4);

    ffio_wfourcc(dyn_ctx, "Xing");

    avio_wb32(dyn_ctx, 0x01 | 0x02 | 0x04 | 0x08);  // frames / size / TOC / vbr scale



    mp3->size = mpah.frame_size;

    mp3->want=1;

    mp3->seen=0;

    mp3->pos=0;



    avio_wb32(dyn_ctx, 0);  // frames

    avio_wb32(dyn_ctx, 0);  // size



    // TOC

    for (i = 0; i < XING_TOC_SIZE; i++)

        avio_w8(dyn_ctx, (uint8_t)(255 * i / XING_TOC_SIZE));



    // vbr quality

    // we write it, because some (broken) tools always expect it to be present

    avio_wb32(dyn_ctx, 0);



    // encoder short version string

    if (enc) {

        uint8_t encoder_str[9] = { 0 };

        if (   strlen(enc->value) > sizeof(encoder_str)

            && !strcmp("Lavc libmp3lame", enc->value)) {

            memcpy(encoder_str, "Lavf lame", 9);

        } else

            memcpy(encoder_str, enc->value, FFMIN(strlen(enc->value), sizeof(encoder_str)));



        avio_write(dyn_ctx, encoder_str, sizeof(encoder_str));

    } else

        avio_write(dyn_ctx, "Lavf\0\0\0\0\0", 9);



    avio_w8(dyn_ctx, 0);      // tag revision 0 / unknown vbr method

    avio_w8(dyn_ctx, 0);      // unknown lowpass filter value

    ffio_fill(dyn_ctx, 0, 8); // empty replaygain fields

    avio_w8(dyn_ctx, 0);      // unknown encoding flags

    avio_w8(dyn_ctx, 0);      // unknown abr/minimal bitrate



    // encoder delay

    if (codec->initial_padding - 528 - 1 >= 1 << 12) {

        av_log(s, AV_LOG_WARNING, "Too many samples of initial padding.\n");

    }

    avio_wb24(dyn_ctx, FFMAX(codec->initial_padding - 528 - 1, 0)<<12);



    avio_w8(dyn_ctx,   0); // misc

    avio_w8(dyn_ctx,   0); // mp3gain

    avio_wb16(dyn_ctx, 0); // preset



    // audio length and CRCs (will be updated later)

    avio_wb32(dyn_ctx, 0); // music length

    avio_wb16(dyn_ctx, 0); // music crc

    avio_wb16(dyn_ctx, 0); // tag crc



    ffio_fill(dyn_ctx, 0, mpah.frame_size - bytes_needed);



    mp3->xing_frame_size   = avio_close_dyn_buf(dyn_ctx, &mp3->xing_frame);

    mp3->xing_frame_offset = avio_tell(s->pb);

    avio_write(s->pb, mp3->xing_frame, mp3->xing_frame_size);



    mp3->audio_size = mp3->xing_frame_size;



    return 0;

}
