static av_cold void init_cavlc_level_tab(void){

    int suffix_length, mask;

    unsigned int i;



    for(suffix_length=0; suffix_length<7; suffix_length++){

        for(i=0; i<(1<<LEVEL_TAB_BITS); i++){

            int prefix= LEVEL_TAB_BITS - av_log2(2*i);

            int level_code= (prefix<<suffix_length) + (i>>(LEVEL_TAB_BITS-prefix-1-suffix_length)) - (1<<suffix_length);



            mask= -(level_code&1);

            level_code= (((2+level_code)>>1) ^ mask) - mask;

            if(prefix + 1 + suffix_length <= LEVEL_TAB_BITS){

                cavlc_level_tab[suffix_length][i][0]= level_code;

                cavlc_level_tab[suffix_length][i][1]= prefix + 1 + suffix_length;

            }else if(prefix + 1 <= LEVEL_TAB_BITS){

                cavlc_level_tab[suffix_length][i][0]= prefix+100;

                cavlc_level_tab[suffix_length][i][1]= prefix + 1;

            }else{

                cavlc_level_tab[suffix_length][i][0]= LEVEL_TAB_BITS+100;

                cavlc_level_tab[suffix_length][i][1]= LEVEL_TAB_BITS;

            }

        }

    }

}
