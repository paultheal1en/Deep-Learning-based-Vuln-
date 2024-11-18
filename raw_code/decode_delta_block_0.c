static int decode_delta_block (bit_buffer_t *bitbuf,

			uint8_t *current, uint8_t *previous, int pitch,

			svq1_pmv_t *motion, int x, int y) {

  uint32_t bit_cache;

  uint32_t block_type;

  int	   result = 0;



  /* get block type */

  bit_cache = get_bit_cache (bitbuf);



  bit_cache	>>= (32 - 3);

  block_type	  = block_type_table[bit_cache].value;

  skip_bits(bitbuf,block_type_table[bit_cache].length);



  /* reset motion vectors */

  if (block_type == SVQ1_BLOCK_SKIP || block_type == SVQ1_BLOCK_INTRA) {

    motion[0].x		  = 0;

    motion[0].y		  = 0;

    motion[(x / 8) + 2].x = 0;

    motion[(x / 8) + 2].y = 0;

    motion[(x / 8) + 3].x = 0;

    motion[(x / 8) + 3].y = 0;

  }



  switch (block_type) {

  case SVQ1_BLOCK_SKIP:

    skip_block (current, previous, pitch, x, y);

    break;



  case SVQ1_BLOCK_INTER:

    result = motion_inter_block (bitbuf, current, previous, pitch, motion, x, y);



    if (result != 0)

    {

#ifdef DEBUG_SVQ1

    printf("Error in motion_inter_block %i\n",result);

#endif

      break;

    }

    result = decode_svq1_block (bitbuf, current, pitch, 0);

    break;



  case SVQ1_BLOCK_INTER_4V:

    result = motion_inter_4v_block (bitbuf, current, previous, pitch, motion, x, y);



    if (result != 0)

    {

#ifdef DEBUG_SVQ1

    printf("Error in motion_inter_4v_block %i\n",result);

#endif

      break;

    }

    result = decode_svq1_block (bitbuf, current, pitch, 0);

    break;



  case SVQ1_BLOCK_INTRA:

    result = decode_svq1_block (bitbuf, current, pitch, 1);

    break;

  }



  return result;

}
