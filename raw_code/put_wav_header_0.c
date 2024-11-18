int put_wav_header(ByteIOContext *pb, AVCodecContext *enc)

{

    int bps, blkalign, bytespersec;

    int hdrsize = 18;



    if(!enc->codec_tag || enc->codec_tag > 0xffff)

       enc->codec_tag = codec_get_tag(codec_wav_tags, enc->codec_id);

    if(!enc->codec_tag)

        return -1;



    put_le16(pb, enc->codec_tag);

    put_le16(pb, enc->channels);

    put_le32(pb, enc->sample_rate);

    if (enc->codec_id == CODEC_ID_PCM_U8 ||

        enc->codec_id == CODEC_ID_PCM_ALAW ||

        enc->codec_id == CODEC_ID_PCM_MULAW) {

        bps = 8;

    } else if (enc->codec_id == CODEC_ID_MP2 || enc->codec_id == CODEC_ID_MP3) {

        bps = 0;

    } else if (enc->codec_id == CODEC_ID_ADPCM_IMA_WAV || enc->codec_id == CODEC_ID_ADPCM_MS || enc->codec_id == CODEC_ID_ADPCM_G726 || enc->codec_id == CODEC_ID_ADPCM_YAMAHA) { //

        bps = 4;

    } else if (enc->codec_id == CODEC_ID_PCM_S24LE) {

        bps = 24;

    } else if (enc->codec_id == CODEC_ID_PCM_S32LE) {

        bps = 32;

    } else {

        bps = 16;

    }



    if (enc->codec_id == CODEC_ID_MP2 || enc->codec_id == CODEC_ID_MP3) {

        blkalign = enc->frame_size; //this is wrong, but seems many demuxers dont work if this is set correctly

        //blkalign = 144 * enc->bit_rate/enc->sample_rate;

    } else if (enc->codec_id == CODEC_ID_ADPCM_G726) { //

        blkalign = 1;

    } else if (enc->block_align != 0) { /* specified by the codec */

        blkalign = enc->block_align;

    } else

        blkalign = enc->channels*bps >> 3;

    if (enc->codec_id == CODEC_ID_PCM_U8 ||

        enc->codec_id == CODEC_ID_PCM_S24LE ||

        enc->codec_id == CODEC_ID_PCM_S32LE ||

        enc->codec_id == CODEC_ID_PCM_S16LE) {

        bytespersec = enc->sample_rate * blkalign;

    } else {

        bytespersec = enc->bit_rate / 8;

    }

    put_le32(pb, bytespersec); /* bytes per second */

    put_le16(pb, blkalign); /* block align */

    put_le16(pb, bps); /* bits per sample */

    if (enc->codec_id == CODEC_ID_MP3) {

        put_le16(pb, 12); /* wav_extra_size */

        hdrsize += 12;

        put_le16(pb, 1); /* wID */

        put_le32(pb, 2); /* fdwFlags */

        put_le16(pb, 1152); /* nBlockSize */

        put_le16(pb, 1); /* nFramesPerBlock */

        put_le16(pb, 1393); /* nCodecDelay */

    } else if (enc->codec_id == CODEC_ID_MP2) {

        put_le16(pb, 22); /* wav_extra_size */

        hdrsize += 22;

        put_le16(pb, 2);  /* fwHeadLayer */

        put_le32(pb, enc->bit_rate); /* dwHeadBitrate */

        put_le16(pb, enc->channels == 2 ? 1 : 8); /* fwHeadMode */

        put_le16(pb, 0);  /* fwHeadModeExt */

        put_le16(pb, 1);  /* wHeadEmphasis */

        put_le16(pb, 16); /* fwHeadFlags */

        put_le32(pb, 0);  /* dwPTSLow */

        put_le32(pb, 0);  /* dwPTSHigh */

    } else if (enc->codec_id == CODEC_ID_ADPCM_IMA_WAV) {

        put_le16(pb, 2); /* wav_extra_size */

        hdrsize += 2;

        put_le16(pb, ((enc->block_align - 4 * enc->channels) / (4 * enc->channels)) * 8 + 1); /* wSamplesPerBlock */

    } else if(enc->extradata_size){

        put_le16(pb, enc->extradata_size);

        put_buffer(pb, enc->extradata, enc->extradata_size);

        hdrsize += enc->extradata_size;

        if(hdrsize&1){

            hdrsize++;

            put_byte(pb, 0);

        }

    } else {

        hdrsize -= 2;

    }



    return hdrsize;

}
