static void put_cabac(CABACContext *c, uint8_t * const state, int bit){

    int RangeLPS= ff_h264_lps_range[2*(c->range&0xC0) + *state];



    if(bit == ((*state)&1)){

        c->range -= RangeLPS;

        *state= ff_h264_mps_state[*state];

    }else{

        c->low += c->range - RangeLPS;

        c->range = RangeLPS;

        *state= ff_h264_lps_state[*state];

    }



    renorm_cabac_encoder(c);

}
