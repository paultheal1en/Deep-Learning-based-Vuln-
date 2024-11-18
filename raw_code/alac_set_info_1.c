static int alac_set_info(ALACContext *alac)

{

    const unsigned char *ptr = alac->avctx->extradata;



    ptr += 4; /* size */

    ptr += 4; /* alac */

    ptr += 4; /* 0 ? */



    if(AV_RB32(ptr) >= UINT_MAX/4){

        av_log(alac->avctx, AV_LOG_ERROR, "setinfo_max_samples_per_frame too large\n");

        return -1;

    }



    /* buffer size / 2 ? */

    alac->setinfo_max_samples_per_frame = bytestream_get_be32(&ptr);

    ptr++;                          /* ??? */

    alac->setinfo_sample_size           = *ptr++;

    alac->setinfo_rice_historymult      = *ptr++;

    alac->setinfo_rice_initialhistory   = *ptr++;

    alac->setinfo_rice_kmodifier        = *ptr++;

    alac->numchannels                   = *ptr++;

    bytestream_get_be16(&ptr);      /* ??? */

    bytestream_get_be32(&ptr);      /* max coded frame size */

    bytestream_get_be32(&ptr);      /* bitrate ? */

    bytestream_get_be32(&ptr);      /* samplerate */



    return 0;

}
