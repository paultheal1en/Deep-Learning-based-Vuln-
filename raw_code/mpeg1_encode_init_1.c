void mpeg1_encode_init(MpegEncContext *s)

{

    static int done=0;

    if(!done){

        int f_code;

        int mv;



        done=1;

        for(f_code=1; f_code<=MAX_FCODE; f_code++){

            for(mv=-MAX_MV; mv<=MAX_MV; mv++){

                int len;



                if(mv==0) len= mbMotionVectorTable[0][1];

                else{

                    int val, bit_size, range, code;



                    bit_size = s->f_code - 1;

                    range = 1 << bit_size;



                    val=mv;

                    if (val < 0) 

                        val = -val;

                    val--;

                    code = (val >> bit_size) + 1;

                    if(code<17){

                        len= mbMotionVectorTable[code][1] + 1 + bit_size;

                    }else{

                        len= mbMotionVectorTable[16][1] + 2 + bit_size;

                    }

                }



                mv_penalty[f_code][mv+MAX_MV]= len;

            }

        }

        



        for(f_code=MAX_FCODE; f_code>0; f_code--){

            for(mv=-(8<<f_code); mv<(8<<f_code); mv++){

                fcode_tab[mv+MAX_MV]= f_code;

            }

        }

    }

    s->mv_penalty= mv_penalty;

    

    s->fcode_tab= fcode_tab;

}
