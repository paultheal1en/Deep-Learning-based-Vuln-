static int split_field_half_ref_list(Picture *dest, int dest_len,

                                     Picture *src,  int src_len,  int parity){

    int same_parity   = 1;

    int same_i        = 0;

    int opp_i         = 0;

    int out_i;

    int field_output;



    for (out_i = 0; out_i < dest_len; out_i += field_output) {

        if (same_parity && same_i < src_len) {

            field_output = split_field_copy(dest + out_i, src + same_i,

                                            parity, 1);

            same_parity = !field_output;

            same_i++;



        } else if (opp_i < src_len) {

            field_output = split_field_copy(dest + out_i, src + opp_i,

                                            PICT_FRAME - parity, 0);

            same_parity = field_output;

            opp_i++;



        } else {

            break;

        }

    }



    return out_i;

}
