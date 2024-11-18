static void selfTest(uint8_t *src[3], int stride[3], int w, int h){

	enum PixelFormat srcFormat, dstFormat;

	int srcW, srcH, dstW, dstH;

	int flags;



	for(srcFormat = 0; srcFormat < PIX_FMT_NB; srcFormat++) {

		for(dstFormat = 0; dstFormat < PIX_FMT_NB; dstFormat++) {

			printf("%s -> %s\n",

					sws_format_name(srcFormat),

					sws_format_name(dstFormat));



			srcW= w;

			srcH= h;

			for(dstW=w - w/3; dstW<= 4*w/3; dstW+= w/3){

				for(dstH=h - h/3; dstH<= 4*h/3; dstH+= h/3){

					for(flags=1; flags<33; flags*=2) {

						int res;



						res = doTest(src, stride, w, h, srcFormat, dstFormat,

							srcW, srcH, dstW, dstH, flags);

						if (res < 0) {

							dstW = 4 * w / 3;

							dstH = 4 * h / 3;

							flags = 33;

						}

					}

				}

			}

		}

	}

}
