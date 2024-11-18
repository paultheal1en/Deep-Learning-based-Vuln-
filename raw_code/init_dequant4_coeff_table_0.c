static void init_dequant4_coeff_table(H264Context *h){

    int i,j,q,x;

    const int transpose = (h->h264dsp.h264_idct_add != ff_h264_idct_add_c); //FIXME ugly

    for(i=0; i<6; i++ ){

        h->dequant4_coeff[i] = h->dequant4_buffer[i];

        for(j=0; j<i; j++){

            if(!memcmp(h->pps.scaling_matrix4[j], h->pps.scaling_matrix4[i], 16*sizeof(uint8_t))){

                h->dequant4_coeff[i] = h->dequant4_buffer[j];

                break;

            }

        }

        if(j<i)

            continue;



        for(q=0; q<52; q++){

            int shift = div6[q] + 2;

            int idx = rem6[q];

            for(x=0; x<16; x++)

                h->dequant4_coeff[i][q][transpose ? (x>>2)|((x<<2)&0xF) : x] =

                    ((uint32_t)dequant4_coeff_init[idx][(x&1) + ((x>>2)&1)] *

                    h->pps.scaling_matrix4[i][x]) << shift;

        }

    }

}
