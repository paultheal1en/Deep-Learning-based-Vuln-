static void init_2d_vlc_rl(RLTable *rl)

{

    int i;

    

    init_vlc(&rl->vlc, TEX_VLC_BITS, rl->n + 2, 

             &rl->table_vlc[0][1], 4, 2,

             &rl->table_vlc[0][0], 4, 2);



    

    rl->rl_vlc[0]= av_malloc(rl->vlc.table_size*sizeof(RL_VLC_ELEM));

    for(i=0; i<rl->vlc.table_size; i++){

        int code= rl->vlc.table[i][0];

        int len = rl->vlc.table[i][1];

        int level, run;

    

        if(len==0){ // illegal code

            run= 65;

            level= MAX_LEVEL;

        }else if(len<0){ //more bits needed

            run= 0;

            level= code;

        }else{

            if(code==rl->n){ //esc

                run= 65;

                level= 0;

            }else if(code==rl->n+1){ //eob

                run= 0;

                level= 127;

            }else{

                run=   rl->table_run  [code] + 1;

                level= rl->table_level[code];

            }

        }

        rl->rl_vlc[0][i].len= len;

        rl->rl_vlc[0][i].level= level;

        rl->rl_vlc[0][i].run= run;

    }

}
