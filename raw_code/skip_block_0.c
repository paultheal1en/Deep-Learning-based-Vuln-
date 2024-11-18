static void skip_block (uint8_t *current, uint8_t *previous, int pitch, int x, int y) {

  uint8_t *src;

  uint8_t *dst;

  int	   i;



  src = &previous[x + y*pitch];

  dst = current;



  for (i=0; i < 16; i++) {

    memcpy (dst, src, 16);

    src += pitch;

    dst += pitch;

  }

}
