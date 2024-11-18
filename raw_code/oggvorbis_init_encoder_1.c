int oggvorbis_init_encoder(vorbis_info *vi, AVCodecContext *avccontext) {

    if(avccontext->coded_frame->quality) /* VBR requested */

	return vorbis_encode_init_vbr(vi, avccontext->channels,

		  avccontext->sample_rate, (float)avccontext->coded_frame->quality / 1000) ;



    return vorbis_encode_init(vi, avccontext->channels,

	          avccontext->sample_rate, -1, avccontext->bit_rate, -1) ;

}
