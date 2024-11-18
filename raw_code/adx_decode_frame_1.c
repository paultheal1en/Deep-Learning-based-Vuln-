static int adx_decode_frame(AVCodecContext *avctx,

			    void *data, int *data_size,

			    const uint8_t *buf0, int buf_size)

{

	ADXContext *c = avctx->priv_data;

	short *samples = data;

	const uint8_t *buf = buf0;

	int rest = buf_size;



	if (!c->header_parsed) {

		int hdrsize = adx_decode_header(avctx,buf,rest);

		if (hdrsize==0) return -1;

		c->header_parsed = 1;

		buf  += hdrsize;

		rest -= hdrsize;

	}



	if (c->in_temp) {

		int copysize = 18*avctx->channels - c->in_temp;

		memcpy(c->dec_temp+c->in_temp,buf,copysize);

		rest -= copysize;

		buf  += copysize;

		if (avctx->channels==1) {

			adx_decode(samples,c->dec_temp,c->prev);

			samples += 32;

		} else {

			adx_decode_stereo(samples,c->dec_temp,c->prev);

			samples += 32*2;

		}

	}

	//

	if (avctx->channels==1) {

		while(rest>=18) {

			adx_decode(samples,buf,c->prev);

			rest-=18;

			buf+=18;

			samples+=32;

		}

	} else {

		while(rest>=18*2) {

			adx_decode_stereo(samples,buf,c->prev);

			rest-=18*2;

			buf+=18*2;

			samples+=32*2;

		}

	}

	//

	c->in_temp = rest;

	if (rest) {

		memcpy(c->dec_temp,buf,rest);

		buf+=rest;

	}

	*data_size = (uint8_t*)samples - (uint8_t*)data;

//	printf("%d:%d ",buf-buf0,*data_size); fflush(stdout);

	return buf-buf0;

}
