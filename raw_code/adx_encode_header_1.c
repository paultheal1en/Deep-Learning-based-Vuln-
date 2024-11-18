static int adx_encode_header(AVCodecContext *avctx,unsigned char *buf,size_t bufsize)

{

#if 0

	struct {

		uint32_t offset; /* 0x80000000 + sample start - 4 */

		unsigned char unknown1[3]; /* 03 12 04 */

		unsigned char channel; /* 1 or 2 */

		uint32_t freq;

		uint32_t size;

		uint32_t unknown2; /* 01 f4 03 00 */

		uint32_t unknown3; /* 00 00 00 00 */

		uint32_t unknown4; /* 00 00 00 00 */



	/* if loop

		unknown3 00 15 00 01

		unknown4 00 00 00 01

		long loop_start_sample;

		long loop_start_byte;

		long loop_end_sample;

		long loop_end_byte;

		long 

	*/

	} adxhdr; /* big endian */

	/* offset-6 "(c)CRI" */

#endif

	write_long(buf+0x00,0x80000000|0x20);

	write_long(buf+0x04,0x03120400|avctx->channels);

	write_long(buf+0x08,avctx->sample_rate);

	write_long(buf+0x0c,0); /* FIXME: set after */

	write_long(buf+0x10,0x01040300);

	write_long(buf+0x14,0x00000000);

	write_long(buf+0x18,0x00000000);

	memcpy(buf+0x1c,"\0\0(c)CRI",8);

	return 0x20+4;

}
