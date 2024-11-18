static av_cold int oggvorbis_encode_close(AVCodecContext *avccontext)

{

    OggVorbisContext *context = avccontext->priv_data;

/*  ogg_packet op ; */



    vorbis_analysis_wrote(&context->vd, 0);  /* notify vorbisenc this is EOF */



    vorbis_block_clear(&context->vb);

    vorbis_dsp_clear(&context->vd);

    vorbis_info_clear(&context->vi);



    av_freep(&avccontext->coded_frame);

    av_freep(&avccontext->extradata);



    return 0;

}
