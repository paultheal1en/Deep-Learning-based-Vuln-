static always_inline void dv_encode_ac(EncBlockInfo* bi, PutBitContext* pb_pool, 

                                       int pb_size)

{

    int run;

    int bits_left;

    PutBitContext* pb = pb_pool;

    int size = bi->partial_bit_count;

    uint32_t vlc = bi->partial_bit_buffer;

    

    bi->partial_bit_count = bi->partial_bit_buffer = 0;

vlc_loop:

       /* Find suitable storage space */

       for (; size > (bits_left = put_bits_left(pb)); pb++) {

          if (bits_left) {

              size -= bits_left;

	      put_bits(pb, bits_left, vlc >> size);

	      vlc = vlc & ((1<<size)-1);

	  }

	  if (pb_size == 1) {

	      bi->partial_bit_count = size;

	      bi->partial_bit_buffer = vlc;

	      return;

	  }

	  --pb_size;

       }

       

       /* Store VLC */

       put_bits(pb, size, vlc);

       

       /* Construct the next VLC */

       run = 0;

       for (; bi->cur_ac < 64; bi->cur_ac++, run++) {

           if (bi->mb[bi->cur_ac]) {

	       size = dv_rl2vlc(run, bi->mb[bi->cur_ac], &vlc);

	       bi->cur_ac++;

	       goto vlc_loop;

	   }

       }

   

       if (bi->cur_ac == 64) {

           size = 4; vlc = 6; /* End Of Block stamp */

	   bi->cur_ac++;

	   goto vlc_loop;

       }

}
