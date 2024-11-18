static inline void RENAME(yvu9toyv12)(const uint8_t *ysrc, const uint8_t *usrc, const uint8_t *vsrc,

	uint8_t *ydst, uint8_t *udst, uint8_t *vdst,

	unsigned int width, unsigned int height, int lumStride, int chromStride)

{

	/* Y Plane */

	memcpy(ydst, ysrc, width*height);



	/* XXX: implement upscaling for U,V */

}
