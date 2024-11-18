static int adx_decode_header(AVCodecContext *avctx,const unsigned char *buf,size_t bufsize)

{

	int offset;

	int channels,freq,size;



	offset = is_adx(buf,bufsize);

	if (offset==0) return 0;



	channels = buf[7];

	freq = read_long(buf+8);

	size = read_long(buf+12);



//	printf("freq=%d ch=%d\n",freq,channels);



	avctx->sample_rate = freq;

	avctx->channels = channels;

	avctx->bit_rate = freq*channels*18*8/32;

//	avctx->frame_size = 18*channels;



	return offset;

}
