static av_cold void init_mv_penalty_and_fcode(MpegEncContext *s)

{

    int f_code;

    int mv;



    for(f_code=1; f_code<=MAX_FCODE; f_code++){

        for(mv=-MAX_MV; mv<=MAX_MV; mv++){

            int len;



            if(mv==0) len= ff_mvtab[0][1];

            else{

                int val, bit_size, code;



                bit_size = f_code - 1;



                val=mv;

                if (val < 0)

                    val = -val;

                val--;

                code = (val >> bit_size) + 1;

                if(code<33){

                    len= ff_mvtab[code][1] + 1 + bit_size;

                }else{

                    len= ff_mvtab[32][1] + av_log2(code>>5) + 2 + bit_size;

                }

            }



            mv_penalty[f_code][mv+MAX_MV]= len;

        }

    }



    for(f_code=MAX_FCODE; f_code>0; f_code--){

        for(mv=-(16<<f_code); mv<(16<<f_code); mv++){

            fcode_tab[mv+MAX_MV]= f_code;

        }

    }



    for(mv=0; mv<MAX_MV*2+1; mv++){

        umv_fcode_tab[mv]= 1;

    }

}
