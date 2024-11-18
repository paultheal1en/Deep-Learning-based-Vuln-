static inline void RENAME(yvu9toyv12)(const uint8_t *ysrc, const uint8_t *usrc, const uint8_t *vsrc,

	uint8_t *ydst, uint8_t *udst, uint8_t *vdst,

	long width, long height, long lumStride, long chromStride)

{

	/* Y Plane */

	memcpy(ydst, ysrc, width*height);



	/* XXX: implement upscaling for U,V */

}
